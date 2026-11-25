"""Primitive and hierarchy definitions for MiniGeom.

Coordinates use one consistent user unit (normally cm for OpenMC models).
Rotations are XYZ Euler angles in degrees and compose as Rz @ Ry @ Rx.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Iterable

import numpy as np
import trimesh


def _as_vec3(values, label: str) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain exactly three finite values")
    return tuple(float(x) for x in arr)


def _rotation_xyz(deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.deg2rad(_as_vec3(deg, "rotation"))
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


def _transform(position, rotation_deg) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _rotation_xyz(rotation_deg)
    matrix[:3, 3] = np.asarray(_as_vec3(position, "position"), dtype=float)
    return matrix


class Primitive:
    """Base class for an analytical region and an optional display mesh."""
    is_bounded: bool = False

    def mesh(self) -> trimesh.Trimesh:
        raise NotImplementedError(f"No preview mesh is available for {type(self).__name__}")

    def analytic(self) -> dict:
        raise NotImplementedError

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Conservative local AABB; None means unbounded or not known."""
        return None


@dataclass
class MeshPrimitive(Primitive):
    """Imported triangle mesh used for visualization and mesh export only.

    MeshPrimitive is deliberately not part of the analytical CSG surface model.
    It can be placed, previewed and exported as STL, but analytical exporters
    (OpenMC and TRIPOLI-5) reject it unless it is converted to supported primitives.
    """
    mesh_data: trimesh.Trimesh
    source: str | None = None
    is_bounded = True

    def __post_init__(self):
        if not isinstance(self.mesh_data, trimesh.Trimesh):
            raise TypeError("MeshPrimitive.mesh_data must be a trimesh.Trimesh")
        if self.mesh_data.is_empty or len(self.mesh_data.faces) == 0:
            raise ValueError("MeshPrimitive requires a non-empty triangular mesh")
        self.mesh_data = self.mesh_data.copy()
        if not np.all(np.isfinite(self.mesh_data.bounds)):
            raise ValueError("MeshPrimitive contains non-finite vertex coordinates")

    @classmethod
    def from_file(cls, filename, *, scale: float = 1.0, process: bool = True,
                  require_watertight: bool = True) -> "MeshPrimitive":
        """Load an STL/mesh file. STL units are not encoded; ``scale`` is explicit."""
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {path}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a finite positive value")
        loaded = trimesh.load(path, force="mesh", process=process)
        if isinstance(loaded, trimesh.Scene):
            geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geometries:
                raise ValueError(f"No triangle mesh was found in: {path}")
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
            raise ValueError(f"The file does not contain a usable triangle mesh: {path}")
        loaded.apply_scale(float(scale))
        if require_watertight and not loaded.is_watertight:
            raise ValueError(
                f"The imported mesh is not watertight: {path}. "
                "Set require_watertight=False only when a non-solid mesh is intentional."
            )
        return cls(mesh_data=loaded, source=str(path.resolve()))

    def mesh(self) -> trimesh.Trimesh:
        return self.mesh_data.copy()

    def analytic(self) -> dict:
        """Return mesh provenance metadata; this is not an analytical surface."""
        return {
            "type": "mesh",
            "source": self.source,
            "vertices": int(len(self.mesh_data.vertices)),
            "faces": int(len(self.mesh_data.faces)),
        }

    def local_bounds(self):
        return self.mesh_data.bounds[0].copy(), self.mesh_data.bounds[1].copy()


@dataclass
class Sphere(Primitive):
    radius: float
    subdivisions: int = 3
    is_bounded = True

    def __post_init__(self):
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Sphere.radius must be finite and strictly positive")
        if not isinstance(self.subdivisions, int) or self.subdivisions < 1:
            raise ValueError("Sphere.subdivisions must be an integer >= 1")

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.creation.icosphere(subdivisions=self.subdivisions, radius=self.radius)

    def analytic(self) -> dict:
        return {"type": "sphere", "radius": float(self.radius)}

    def local_bounds(self):
        extent = np.full(3, self.radius, dtype=float)
        return -extent, extent


@dataclass
class Cylinder(Primitive):
    """Finite cylinder centered at origin, with its axis along local Z."""
    radius: float
    height: float
    sections: int = 64
    is_bounded = True

    def __post_init__(self):
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Cylinder.radius must be finite and strictly positive")
        if not math.isfinite(self.height) or self.height <= 0:
            raise ValueError("Cylinder.height must be finite and strictly positive")
        if not isinstance(self.sections, int) or self.sections < 8:
            raise ValueError("Cylinder.sections must be an integer >= 8")

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(radius=self.radius, height=self.height, sections=self.sections)

    def analytic(self) -> dict:
        return {"type": "cylinder", "radius": float(self.radius),
                "height": float(self.height), "axis": [0., 0., 1.]}

    def local_bounds(self):
        h = np.array([self.radius, self.radius, self.height / 2.])
        return -h, h


@dataclass
class Box(Primitive):
    x: float
    y: float
    z: float
    is_bounded = True

    def __post_init__(self):
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Box.{name} must be finite and strictly positive")

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=(self.x, self.y, self.z))

    def analytic(self) -> dict:
        return {"type": "box", "size": [float(self.x), float(self.y), float(self.z)]}

    def local_bounds(self):
        h = np.array([self.x, self.y, self.z], dtype=float) / 2
        return -h, h


@dataclass
class Cone(Primitive):
    """Finite conical frustum along local Z, radii specified at bottom/top."""
    bottom_radius: float
    top_radius: float
    height: float
    sections: int = 64
    is_bounded = True

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.bottom_radius, self.top_radius)):
            raise ValueError("Cone radii must be finite and >= 0")
        if self.bottom_radius == 0 and self.top_radius == 0:
            raise ValueError("At least one cone radius must be strictly positive")
        if not math.isfinite(self.height) or self.height <= 0:
            raise ValueError("Cone.height must be finite and strictly positive")
        if not isinstance(self.sections, int) or self.sections < 8:
            raise ValueError("Cone.sections must be an integer >= 8")

    def mesh(self) -> trimesh.Trimesh:
        n = self.sections
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        z0, z1 = -self.height / 2.0, self.height / 2.0
        bottom = np.column_stack((self.bottom_radius * np.cos(angles),
                                  self.bottom_radius * np.sin(angles),
                                  np.full(n, z0)))
        top = np.column_stack((self.top_radius * np.cos(angles),
                               self.top_radius * np.sin(angles),
                               np.full(n, z1)))
        vertices = np.vstack((bottom, top, [[0., 0., z0], [0., 0., z1]]))
        cb, ct = 2*n, 2*n+1
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.extend(((i, j, n+j), (i, n+j, n+i)))
            faces.append((cb, j, i))
            faces.append((ct, n+i, n+j))
        return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)

    def analytic(self) -> dict:
        return {"type": "cone", "bottom_radius": float(self.bottom_radius),
                "top_radius": float(self.top_radius), "height": float(self.height),
                "axis": "z"}

    def local_bounds(self):
        r = max(self.bottom_radius, self.top_radius)
        h = np.array([r, r, self.height / 2.])
        return -h, h


@dataclass
class Ellipsoid(Primitive):
    """Ellipsoid centered at the local origin with semi-axes (rx, ry, rz)."""
    radii: tuple[float, float, float]
    subdivisions: int = 3
    is_bounded = True

    def __post_init__(self):
        self.radii = _as_vec3(self.radii, "radii")
        if any(v <= 0 for v in self.radii):
            raise ValueError("Ellipsoid semi-axes must be strictly positive")
        if not isinstance(self.subdivisions, int) or self.subdivisions < 1:
            raise ValueError("Ellipsoid.subdivisions must be an integer >= 1")

    def mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.icosphere(subdivisions=self.subdivisions, radius=1.0)
        mesh.apply_scale(self.radii)
        return mesh

    def analytic(self) -> dict:
        return {"type": "ellipsoid", "radii": list(self.radii)}

    def local_bounds(self):
        e = np.asarray(self.radii, dtype=float)
        return -e, e


@dataclass
class Torus(Primitive):
    """Solid ring torus; its symmetry axis is local x, y or z."""
    major_radius: float
    minor_radius: float
    axis: str = "z"
    major_sections: int = 64
    minor_sections: int = 24
    is_bounded = True

    def __post_init__(self):
        if not math.isfinite(self.major_radius) or not math.isfinite(self.minor_radius):
            raise ValueError("Torus radii must be finite")
        if self.major_radius <= self.minor_radius or self.minor_radius <= 0:
            raise ValueError("Torus requires major_radius > minor_radius > 0 (ring torus)")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("Torus.axis must be 'x', 'y' or 'z'")
        if self.major_sections < 8 or self.minor_sections < 8:
            raise ValueError("Torus subdivisions must be >= 8")

    def mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.torus(self.major_radius, self.minor_radius,
                                     self.major_sections, self.minor_sections)
        if self.axis == "x":
            mesh.apply_transform(_transform((0, 0, 0), (0, 90, 0)))
        elif self.axis == "y":
            mesh.apply_transform(_transform((0, 0, 0), (90, 0, 0)))
        return mesh

    def analytic(self) -> dict:
        return {"type": "torus", "major_radius": float(self.major_radius),
                "minor_radius": float(self.minor_radius), "axis": self.axis}

    def local_bounds(self):
        r, a = self.major_radius + self.minor_radius, self.minor_radius
        if self.axis == "x":
            e = np.array([a, r, r])
        elif self.axis == "y":
            e = np.array([r, a, r])
        else:
            e = np.array([r, r, a])
        return -e, e


@dataclass
class Plane(Primitive):
    """Infinite half-space bounded by n·x = offset in local coordinates.

    ``sense=-1`` selects n·x-offset <= 0; ``sense=+1`` selects >= 0.
    It is not a bounded solid and therefore cannot be the world volume.
    """
    normal: tuple[float, float, float]
    offset: float = 0.0
    sense: int = -1
    is_bounded = False

    def __post_init__(self):
        n = np.asarray(_as_vec3(self.normal, "normal"), dtype=float)
        norm = float(np.linalg.norm(n))
        if norm == 0:
            raise ValueError("Plane.normal cannot be zero")
        if not math.isfinite(self.offset):
            raise ValueError("Plane.offset must be finite")
        # Keep the plane equation invariant when normalizing n: n.x = offset.
        self.offset = float(self.offset) / norm
        self.normal = tuple(float(x) for x in (n / norm))
        if self.sense not in (-1, 1):
            raise ValueError("Plane.sense must be -1 or +1")

    def analytic(self) -> dict:
        return {"type": "plane", "normal": list(self.normal),
                "offset": float(self.offset), "sense": self.sense}


@dataclass
class Quadric(Primitive):
    """General implicit quadric half-space.

    Coefficient order is (A,B,C,D,E,F,G,H,J,K) for
    Ax²+By²+Cz²+Dxy+Eyz+Fxz+Gx+Hy+Jz+K = 0.
    ``bounded=True`` is an explicit user assertion; MiniGeom does not attempt
    to infer boundedness for arbitrary quadrics.
    """
    coefficients: tuple[float, ...]
    sense: int = -1
    bounded: bool = False
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    is_bounded = False

    def __post_init__(self):
        self.coefficients = tuple(float(v) for v in self.coefficients)
        if len(self.coefficients) != 10 or not all(math.isfinite(v) for v in self.coefficients):
            raise ValueError("Quadric.coefficients must contain 10 finite values")
        if not any(abs(v) > 0 for v in self.coefficients):
            raise ValueError("A quadric cannot have all-zero coefficients")
        if self.sense not in (-1, 1):
            raise ValueError("Quadric.sense must be -1 or +1")
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise ValueError("Quadric.bounds must be a (lower, upper) pair")
            lower = np.asarray(_as_vec3(self.bounds[0], "bounds.lower"), dtype=float)
            upper = np.asarray(_as_vec3(self.bounds[1], "bounds.upper"), dtype=float)
            if np.any(upper <= lower):
                raise ValueError("Quadric.bounds must satisfy upper > lower on every axis")
            self.bounds = (tuple(lower), tuple(upper))
        if self.bounded and self.bounds is None:
            raise ValueError("Quadric.bounded=True also requires an explicit AABB in bounds")
        self.is_bounded = bool(self.bounded and self.bounds is not None)

    def local_bounds(self):
        if self.bounds is None:
            return None
        return np.asarray(self.bounds[0], dtype=float), np.asarray(self.bounds[1], dtype=float)

    def analytic(self) -> dict:
        return {"type": "quadric", "coefficients": list(self.coefficients),
                "sense": self.sense, "bounded_assertion": self.bounded,
                "bounds": [list(v) for v in self.bounds] if self.bounds else None}


@dataclass
class Placement:
    child: "VolumeDef"
    copy_no: int = 1
    name: str | None = None
    position: tuple[float, float, float] = (0., 0., 0.)
    rotation: tuple[float, float, float] = (0., 0., 0.)
    priority: int = 0
    material_override: str | None = None
    can_overlap: bool = False
    overlap_priority: int = 0
    _order: int = field(default=0, repr=False)

    def __post_init__(self):
        if not isinstance(self.child, VolumeDef):
            raise TypeError("Placement.child must be a VolumeDef")
        self.position = _as_vec3(self.position, "position")
        self.rotation = _as_vec3(self.rotation, "rotation")
        if not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        if not isinstance(self.overlap_priority, int):
            raise TypeError("overlap_priority must be an integer")
        if self.material_override is not None:
            if not isinstance(self.material_override, str) or not self.material_override.strip():
                raise ValueError("material_override must be a non-empty string or None")
            self.material_override = self.material_override.strip()
        if self.name is None:
            self.name = f"{self.child.name}#{self.copy_no}"
        if not self.name or "/" in self.name:
            raise ValueError("Placement name must be non-empty and must not contain '/'")

    def local_matrix(self) -> np.ndarray:
        return _transform(self.position, self.rotation)

    def clone(self) -> "Placement":
        return Placement(child=self.child.clone(copy_children=True), copy_no=self.copy_no,
                         name=self.name, position=self.position, rotation=self.rotation,
                         priority=self.priority, material_override=self.material_override,
                         can_overlap=self.can_overlap, overlap_priority=self.overlap_priority,
                         _order=self._order)


@dataclass
class LatticePlacement:
    """Explicit 3-D rectangular lattice of one reusable VolumeDef.

    ``shape`` is (nx, ny, nz), ``lower_left`` is the lower corner in the
    parent's local frame, and ``pitch`` is (px, py, pz). For reliable OpenMC
    transport the child definition must be a Box exactly matching the pitch;
    this gives each lattice element a complete tile universe.
    """
    child: "VolumeDef"
    shape: tuple[int, int, int]
    pitch: tuple[float, float, float]
    lower_left: tuple[float, float, float]
    priority: int = 0
    name: str = "lattice"
    material_override: str | None = None
    _order: int = field(default=0, repr=False)

    def __post_init__(self):
        if not isinstance(self.child, VolumeDef):
            raise TypeError("LatticePlacement.child must be a VolumeDef")
        raw_shape = tuple(self.shape)
        if len(raw_shape) != 3:
            raise ValueError("LatticePlacement.shape must be (nx, ny, nz) with positive values")
        try:
            converted_shape = tuple(int(v) for v in raw_shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("LatticePlacement.shape must contain three positive integers") from exc
        if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(float(v))
               or float(v) != int(v) or int(v) <= 0
               for v in raw_shape):
            raise ValueError("LatticePlacement.shape must contain three positive integers")
        self.shape = converted_shape
        self.pitch = _as_vec3(self.pitch, "pitch")
        if any(v <= 0 for v in self.pitch):
            raise ValueError("Lattice pitches must be positive")
        self.lower_left = _as_vec3(self.lower_left, "lower_left")
        if not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        if self.material_override is not None:
            if not isinstance(self.material_override, str) or not self.material_override.strip():
                raise ValueError("material_override must be a non-empty string or None")
            self.material_override = self.material_override.strip()
        if not self.name or "/" in self.name:
            raise ValueError("Lattice name must be non-empty and must not contain '/'")
        if not isinstance(self.child.shape, Box):
            raise ValueError("The root volume of a lattice element must be a Box")
        expected = np.asarray(self.pitch)
        actual = np.array([self.child.shape.x, self.child.shape.y, self.child.shape.z])
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("The lattice element root Box must exactly match the lattice pitch")

    @property
    def upper_right(self) -> tuple[float, float, float]:
        return tuple(self.lower_left[i] + self.shape[i] * self.pitch[i] for i in range(3))

    def center(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        return tuple(self.lower_left[a] + (idx + 0.5) * self.pitch[a]
                     for a, idx in enumerate((i, j, k)))

    def clone(self) -> "LatticePlacement":
        return LatticePlacement(self.child.clone(copy_children=True), self.shape,
                                self.pitch, self.lower_left, self.priority,
                                self.name, self.material_override, self._order)


_MATERIAL_UNSET = object()


class VolumeDef:
    """Reusable volume definition, analogous to a ROOT TGeoVolume.

    ``material`` is an optional symbolic tag (e.g. ``"steel"``). MiniGeom
    does not define physical material composition; that belongs to a backend.
    ``None`` means that no material tag has been assigned.
    """

    def __init__(self, name: str, shape: Primitive | None = None,
                 material: str | None = None, self_priority: int = -100):
        if not name:
            raise ValueError("VolumeDef.name cannot be empty")
        if shape is not None and not isinstance(shape, Primitive):
            raise TypeError("shape must be a Primitive or None")
        self.name = str(name)
        self.shape = shape
        self.material = self._normalize_material_tag(material)
        self.self_priority = int(self_priority)
        self.children: list[Placement] = []
        self.lattices: list[LatticePlacement] = []
        self._next_order = 0

    @staticmethod
    def _normalize_material_tag(material: str | None) -> str | None:
        if material is None:
            return None
        if not isinstance(material, str):
            raise TypeError("material must be a string or None")
        value = material.strip()
        return value or None

    def _contains_in_descendants(self, target: "VolumeDef", visited=None) -> bool:
        visited = set() if visited is None else visited
        if id(self) in visited:
            return False
        visited.add(id(self))
        for item in [*self.children, *self.lattices]:
            if item.child is target or item.child._contains_in_descendants(target, visited):
                return True
        return False

    def _validate_child_name(self, name: str):
        existing = [p.name for p in self.children] + [p.name for p in self.lattices]
        if name in existing:
            raise ValueError(f"A placement or lattice named '{name}' already exists in '{self.name}'")

    def place(self, child: "VolumeDef", at=(0., 0., 0.), rotate=(0., 0., 0.),
              priority: int = 0, copy_no: int | None = None,
              name: str | None = None, material_override: str | None = None) -> Placement:
        if not isinstance(child, VolumeDef):
            raise TypeError("child must be a VolumeDef")
        if child is self or child._contains_in_descendants(self):
            raise ValueError(f"Cycle detected: cannot place '{child.name}' inside '{self.name}'")
        if not isinstance(priority, int):
            raise TypeError("priority must be an integer")
        if copy_no is None:
            copies = [p.copy_no for p in self.children if p.child is child]
            copy_no = max(copies, default=0) + 1
        placement = Placement(child=child, copy_no=int(copy_no), name=name,
                              position=at, rotation=rotate, priority=priority,
                              material_override=material_override, _order=self._next_order)
        self._validate_child_name(placement.name)
        self._next_order += 1
        self.children.append(placement)
        return placement

    def place_protruding(self, child: "VolumeDef", at=(0., 0., 0.),
                         rotate=(0., 0., 0.), *, overlap_priority: int = 100,
                         local_priority: int = 0, copy_no: int | None = None,
                         name: str | None = None,
                         material_override: str | None = None) -> Placement:
        if not isinstance(overlap_priority, int):
            raise TypeError("overlap_priority must be an integer")
        placement = self.place(child, at=at, rotate=rotate, priority=local_priority,
                               copy_no=copy_no, name=name,
                               material_override=material_override)
        placement.can_overlap = True
        placement.overlap_priority = overlap_priority
        return placement

    place_overlapping = place_protruding

    def place_lattice(self, child: "VolumeDef", *, shape: tuple[int, int, int],
                      pitch: tuple[float, float, float],
                      lower_left: tuple[float, float, float],
                      priority: int = 0, name: str | None = None,
                      material_override: str | None = None) -> LatticePlacement:
        """Add a regular 3-D lattice. ``shape`` uses natural (nx, ny, nz) order."""
        if child is self or child._contains_in_descendants(self):
            raise ValueError(f"Cycle detected: cannot place '{child.name}' inside '{self.name}'")
        lattice = LatticePlacement(
            child=child, shape=shape, pitch=pitch, lower_left=lower_left,
            priority=priority, name=name or f"{child.name}_lattice_{self._next_order}",
            material_override=material_override, _order=self._next_order,
        )
        self._validate_child_name(lattice.name)
        self._next_order += 1
        self.lattices.append(lattice)
        return lattice

    def ordered_children(self):
        return sorted(self.children, key=lambda p: (p.priority, p._order), reverse=True)

    def export_stl(self, filename: str | Path) -> Path:
        """Export this definition's local primitive mesh without its placements."""
        if self.shape is None:
            raise ValueError(f"Volume '{self.name}' has no primitive to export")
        mesh = self.shape.mesh()
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        return path

    def clone(self, name: str | None = None, *, material=_MATERIAL_UNSET,
              self_priority: int | None = None, copy_children: bool = True) -> "VolumeDef":
        # Deep-copy the full graph in one operation. deepcopy's memo preserves
        # shared VolumeDef references (e.g. two placements of the same pin).
        cloned = deepcopy(self) if copy_children else VolumeDef(
            name=self.name, shape=deepcopy(self.shape), material=self.material,
            self_priority=self.self_priority,
        )
        cloned.name = name if name is not None else f"{self.name}_clone"
        if material is not _MATERIAL_UNSET:
            cloned.material = self._normalize_material_tag(material)
        if self_priority is not None:
            cloned.self_priority = int(self_priority)
        return cloned


@dataclass(frozen=True)
class VolumeInstance:
    path: tuple[str, ...]
    volume_def: VolumeDef
    material: str | None
    world_matrix: np.ndarray
    container_path: tuple[str, ...]
    local_priority: int
    local_order: int
    global_order: int
    can_overlap: bool = False
    overlap_priority: int = 0
    placement: Placement | None = None
    lattice_name: str | None = None
    lattice_index: tuple[int, int, int] | None = None

    @property
    def label(self) -> str:
        return "/".join(self.path)

    def world_mesh(self) -> trimesh.Trimesh | None:
        if self.volume_def.shape is None:
            return None
        try:
            mesh = self.volume_def.shape.mesh().copy()
        except NotImplementedError:
            return None
        mesh.apply_transform(self.world_matrix)
        return mesh
