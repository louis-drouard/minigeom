"""Backend-independent analytical CSG compiler and region model."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import (
    Box, Cone, Cylinder, Ellipsoid, MeshPrimitive, Plane, Quadric, Sphere, Torus,
    VolumeInstance,
)


def _quadric_to_matrix(coefficients):
    a, b, c, d, e, f, g, h, j, k = coefficients
    matrix = np.array([[a, d/2, f/2], [d/2, b, e/2], [f/2, e/2, c]], dtype=float)
    linear = np.array([g, h, j], dtype=float)
    return matrix, linear, float(k)


def _matrix_to_quadric(matrix, linear, constant):
    return (
        float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]),
        float(2*matrix[0, 1]), float(2*matrix[1, 2]), float(2*matrix[0, 2]),
        float(linear[0]), float(linear[1]), float(linear[2]), float(constant),
    )


def transform_quadric(coefficients, rotation, translation):
    """Transform a local implicit quadric ``f(x_local) = 0`` into world coordinates.

    A quadric is represented as

        f(x) = x.T @ A @ x + b.T @ x + k.

    For a rigid transform ``x_world = R @ x_local + t``, points in local
    coordinates are ``x_local = R.T @ (x_world - t)``. Substituting that
    relation into the implicit equation gives a new matrix, linear vector and
    constant term in world coordinates. No vertices are sampled and no mesh
    is involved: the transformed surface remains the exact same quadric.
    """
    a, b, k = _quadric_to_matrix(coefficients)
    r = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    aw = r @ a @ r.T
    bw = -2.0 * aw @ t + r @ b
    kw = float(t @ aw @ t - (r @ b) @ t + k)
    return _matrix_to_quadric(aw, bw, kw)


@dataclass
class SurfaceSpec:
    id: int
    name: str
    kind: str
    coefficients: tuple[float, ...]
    boundary_type: str = "transmission"

    def evaluate(self, point) -> float:
        x, y, z = (float(v) for v in point)
        c = self.coefficients
        if self.kind == "sphere":
            x0, y0, z0, r = c
            return (x-x0)**2 + (y-y0)**2 + (z-z0)**2 - r*r
        if self.kind == "plane":
            a, b, cc, d = c
            return a*x + b*y + cc*z - d
        if self.kind == "quadric":
            a,b,cc,d,e,f,g,h,j,k = c
            return (a*x*x + b*y*y + cc*z*z + d*x*y + e*y*z + f*x*z
                    + g*x + h*y + j*z + k)
        if self.kind == "torus":
            x0,y0,z0,major,minor,ux,uy,uz = c
            delta = np.array([x-x0, y-y0, z-z0])
            axis = np.array([ux, uy, uz])
            axial = float(np.dot(delta, axis))
            radial = float(np.linalg.norm(delta - axial*axis))
            return (axial/minor)**2 + ((radial-major)/minor)**2 - 1.0
        raise ValueError(f"Unknown surface type: {self.kind}")

    def to_dict(self) -> dict:
        if self.kind == "sphere":
            x0, y0, z0, r = self.coefficients
            data = {"x0": x0, "y0": y0, "z0": z0, "r": r}
        elif self.kind == "plane":
            a, b, c, d = self.coefficients
            data = {"a": a, "b": b, "c": c, "d": d}
        elif self.kind == "torus":
            x0,y0,z0,major,minor,ux,uy,uz = self.coefficients
            data = {"center": [x0,y0,z0], "major_radius": major,
                    "minor_radius": minor, "axis": [ux,uy,uz]}
        else:
            keys = ("a", "b", "c", "d", "e", "f", "g", "h", "j", "k")
            data = dict(zip(keys, self.coefficients))
        return {"id": self.id, "name": self.name, "type": self.kind,
                "coefficients": data, "boundary_type": self.boundary_type}


class SurfaceRegistry:
    """Deduplicate analytical surfaces using their normalized equations.

    A surface is an implicit equation ``f(x, y, z) = 0``. The registry stores
    one ``SurfaceSpec`` for each unique equation (after rounding coefficients
    to the configured precision). Region expressions then refer to that shared
    surface by half-space sign. This is important for adjacent cells: if two
    cells use the same mathematical boundary, they should reference one surface
    ID rather than two almost-identical copies.

    The registry does not triangulate geometry and does not perform Boolean
    operations. It only owns the analytical boundary definitions and IDs.
    """
    def __init__(self, precision: int = 13):
        if not isinstance(precision, int) or precision < 6:
            raise ValueError("surface_precision must be an integer >= 6")
        self.precision = precision
        self._by_key: dict[tuple, SurfaceSpec] = {}
        self._next_id = 1

    def _get(self, kind: str, coefficients: Iterable[float], name: str) -> SurfaceSpec:
        coeffs = tuple(float(v) for v in coefficients)
        if not all(math.isfinite(v) for v in coeffs):
            raise ValueError("Surface coefficients must be finite")
        key = (kind,) + tuple(round(v, self.precision) for v in coeffs)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        surface = SurfaceSpec(self._next_id, name, kind, coeffs)
        self._next_id += 1
        self._by_key[key] = surface
        return surface

    def sphere(self, center, radius: float, name: str) -> SurfaceSpec:
        return self._get("sphere", (*center, radius), name)

    def plane(self, normal, point, name: str) -> SurfaceSpec:
        n = np.asarray(normal, dtype=float)
        norm = float(np.linalg.norm(n))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("Plane normal cannot be zero")
        n /= norm
        d = float(np.dot(n, np.asarray(point, dtype=float)))
        return self._get("plane", (*n, d), name)

    def quadric(self, coefficients, name: str) -> SurfaceSpec:
        return self._get("quadric", coefficients, name)

    def torus(self, center, major, minor, axis, name):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        return self._get("torus", (*center, major, minor, *axis), name)

    def mark_boundary(self, surfaces: Iterable[SurfaceSpec], boundary_type: str):
        for surface in surfaces:
            surface.boundary_type = boundary_type

    @property
    def surfaces(self) -> list[SurfaceSpec]:
        return sorted(self._by_key.values(), key=lambda surface: surface.id)


class Region:
    """Base class for an analytical set of points in 3-D space.

    Concrete regions are expressions over implicit surfaces:

    * ``Halfspace`` selects one side of a surface;
    * ``Intersection`` selects points satisfying every child region;
    * ``UnionRegion`` selects points satisfying at least one child region;
    * ``Complement`` selects points not contained in its child region.

    These are set operations, not mesh booleans. They remain analytical all
    the way to the OpenMC and AGORA exporters.
    """
    def __and__(self, other: "Region") -> "Region":
        return Intersection((self, other))

    def __or__(self, other: "Region") -> "Region":
        return UnionRegion((self, other))

    def __invert__(self) -> "Region":
        return Complement(self)

    def contains(self, point, tolerance: float = 1e-10) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class Halfspace(Region):
    surface: SurfaceSpec
    sense: int

    def __post_init__(self):
        if self.sense not in (-1, 1):
            raise ValueError("Halfspace.sense must be -1 or +1")

    def contains(self, point, tolerance: float = 1e-10) -> bool:
        # SurfaceSpec.evaluate() returns the signed implicit-function value
        # f(x). A negative half-space is f(x) <= 0; a positive half-space is
        # f(x) >= 0. The small tolerance treats points numerically close to a
        # boundary as belonging to either neighbouring cell for diagnostics.
        value = self.surface.evaluate(point)
        return value <= tolerance if self.sense < 0 else value >= -tolerance

    def to_dict(self) -> dict:
        return {"op": "halfspace", "surface": self.surface.id, "sense": self.sense}


@dataclass(frozen=True)
class Intersection(Region):
    terms: tuple[Region, ...]

    def contains(self, point, tolerance: float = 1e-10) -> bool:
        # Intersection is the set-theoretic AND: a point belongs only when it
        # lies in every operand. Geometrically this is how finite primitives
        # are closed with several independent side/cap constraints.
        return all(term.contains(point, tolerance) for term in self.terms)

    def to_dict(self) -> dict:
        return {"op": "intersection", "terms": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True)
class UnionRegion(Region):
    terms: tuple[Region, ...]

    def contains(self, point, tolerance: float = 1e-10) -> bool:
        # Union is the set-theoretic OR: a point belongs when it is inside at
        # least one operand. It is useful for disconnected but same-material
        # parts, such as the 37 separate fuel pellets in the CANDU example.
        return any(term.contains(point, tolerance) for term in self.terms)

    def to_dict(self) -> dict:
        return {"op": "union", "terms": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True)
class Complement(Region):
    term: Region

    def contains(self, point, tolerance: float = 1e-10) -> bool:
        # Complement reverses membership. It is generally unbounded, which is
        # why CSGCompiler clips user-authored cell regions to the finite domain
        # of their owning VolumeDef before exporting them to a transport code.
        return not self.term.contains(point, tolerance)

    def to_dict(self) -> dict:
        return {"op": "complement", "term": self.term.to_dict()}


def region_union(regions: Iterable[Region | None]) -> Region | None:
    terms = tuple(region for region in regions if region is not None)
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return UnionRegion(terms)


def region_intersection(a: Region | None, b: Region | None) -> Region | None:
    if a is None or b is None:
        return None
    return Intersection((a, b))


def region_subtract(base: Region | None, cutter: Region | None) -> Region | None:
    if base is None:
        return None
    if cutter is None:
        return base
    return Intersection((base, Complement(cutter)))


@dataclass
class CellSpec:
    id: int
    name: str
    path: str
    material: str | None
    region: Region
    priority: int
    volume_name: str
    protruding: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "path": self.path,
                "material": self.material, "priority": self.priority,
                "volume_definition": self.volume_name,
                "protruding": self.protruding, "region": self.region.to_dict()}


@dataclass
class RawInstanceSpec:
    path: str
    region: Region
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    material: str | None
    parent_path: str | None
    priority: int
    order: int
    protruding: bool = False
    lattice_name: str | None = None
    lattice_index: tuple[int, int, int] | None = None

    def to_dict(self):
        return {"path": self.path, "bounds": [list(self.bounds[0]), list(self.bounds[1])],
                "material": self.material, "parent_path": self.parent_path,
                "priority": self.priority, "order": self.order,
                "protruding": self.protruding,
                "lattice_name": self.lattice_name,
                "lattice_index": self.lattice_index,
                "region": self.region.to_dict()}


@dataclass
class CSGModel:
    scene_name: str
    top_volume: str
    surfaces: list[SurfaceSpec]
    cells: list[CellSpec]
    raw_instances: list[RawInstanceSpec] | None = None
    world_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    overlap_rule: str = "local: higher priority wins, then later placement; protruding: higher overlap_priority wins, then later instance"

    def to_dict(self) -> dict:
        return {
            "schema": "minigeom-csg-1.2",
            "format_note": "Intermediate JSON representation. OpenMC native geometry is exported as geometry.xml.",
            "material_semantics": (
                "material is an optional symbolic tag; null means unassigned. "
                "MiniGeom does not define physical material compositions."
            ),
            "scene": self.scene_name, "top_volume": self.top_volume,
            "overlap_rule": self.overlap_rule,
            "world_bounds": self.world_bounds,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "instances": [item.to_dict() for item in (self.raw_instances or [])],
            "cells": [cell.to_dict() for cell in self.cells],
        }

    def export_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def find_cells(self, point, tolerance: float = 1e-10) -> list[CellSpec]:
        """Return every resolved cell containing ``point``, regardless of material tag."""
        return [cell for cell in self.cells if cell.region.contains(point, tolerance)]

    def find_material(self, point, tolerance: float = 1e-10) -> list[CellSpec]:
        """Backward-compatible alias for :meth:`find_cells`.

        The historical name is retained, but membership is geometric and does
        not require a material tag.
        """
        return self.find_cells(point, tolerance)


class CSGCompiler:
    """Compile a Scene into disjoint, world-coordinate analytical cells."""
    def __init__(self, boundary_type: str = "vacuum", surface_precision: int = 13):
        supported = {"vacuum", "reflective", "white", "transmission"}
        if boundary_type not in supported:
            raise ValueError(f"Unsupported boundary_type={boundary_type!r}; periodic boundaries require an explicit pairing")
        self.boundary_type = boundary_type
        self.surface_precision = surface_precision
        self.registry = SurfaceRegistry(surface_precision)
        self._shape_regions = {}
        self._children = {}
        self._cells: list[CellSpec] = []
        self._raw_instances: list[RawInstanceSpec] = []
        self._current_world_bounds = None

    @staticmethod
    def _unit(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Transformation contains a zero or non-finite vector")
        return vector / norm

    def primitive_region(self, shape, matrix: np.ndarray, name: str) -> tuple[Region | None, list[SurfaceSpec]]:
        """Return a primitive's region after applying a homogeneous transform."""
        if shape is None:
            return None, []
        rotation = np.asarray(matrix[:3, :3], dtype=float)
        translation = np.asarray(matrix[:3, 3], dtype=float)
        surfaces: list[SurfaceSpec] = []

        if isinstance(shape, MeshPrimitive):
            raise NotImplementedError(
                "MeshPrimitive is a tessellated display/import object and cannot be "
                "compiled to analytical CSG. Use an analytical primitive for OpenMC "
                "or TRIPOLI-5 export, or export the mesh as STL."
            )

        if isinstance(shape, Sphere):
            # Sphere boundary: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 - R^2 = 0.
            # The sphere interior is the negative half-space of that equation.
            surface = self.registry.sphere(translation, shape.radius, f"{name}_sphere")
            surfaces.append(surface)
            return Halfspace(surface, -1), surfaces

        if isinstance(shape, Cylinder):
            # A finite cylinder is the intersection of three constraints:
            #   1) inside the infinite circular cylinder;
            #   2) before the upper end-cap plane;
            #   3) after the lower end-cap plane.
            #
            # Let a be the transformed unit axis and C a point on that axis.
            # P = I - a*a.T projects a vector onto the plane perpendicular to
            # the axis. The squared distance to the cylinder axis is therefore
            #   (x-C).T @ P @ (x-C).
            # Setting that distance equal to R^2 yields a general quadric:
            #   x.T P x - 2(P C).T x + C.T P C - R^2 = 0.
            # This representation works for cylinders at arbitrary rotations.
            axis = self._unit(rotation @ np.array([0., 0., 1.]))
            projector = np.eye(3) - np.outer(axis, axis)
            linear = -2.0 * (projector @ translation)
            constant = float(translation @ projector @ translation - shape.radius**2)
            coefficients = (projector[0,0], projector[1,1], projector[2,2],
                            2*projector[0,1], 2*projector[1,2], 2*projector[0,2],
                            linear[0], linear[1], linear[2], constant)
            side = self.registry.quadric(coefficients, f"{name}_cylinder_side")
            upper = self.registry.plane(axis, translation + axis*shape.height/2, f"{name}_cap_plus")
            lower = self.registry.plane(axis, translation - axis*shape.height/2, f"{name}_cap_minus")
            surfaces.extend((side, upper, lower))
            # For both cap planes the positive normal points along the cylinder
            # axis. The inside is consequently the negative side of the upper
            # plane and the positive side of the lower plane.
            return Intersection((Halfspace(side, -1), Halfspace(upper, -1), Halfspace(lower, 1))), surfaces

        if isinstance(shape, Box):
            # A box is not exported as a mesh or as a six-face polyhedron
            # object. Each pair of parallel planes defines one bounded slab.
            # Intersecting the X, Y and Z slabs gives the exact box interior.
            half = np.array([shape.x, shape.y, shape.z], dtype=float) / 2
            terms = []
            for i, local_axis in enumerate(np.eye(3)):
                axis = self._unit(rotation @ local_axis)
                upper = self.registry.plane(axis, translation + axis*half[i], f"{name}_axis{i}_plus")
                lower = self.registry.plane(axis, translation - axis*half[i], f"{name}_axis{i}_minus")
                surfaces.extend((upper, lower))
                # The upper face contributes f<=0, while the lower face
                # contributes f>=0, because both normals point along +axis.
                terms.extend((Halfspace(upper, -1), Halfspace(lower, 1)))
            return Intersection(tuple(terms)), surfaces

        if isinstance(shape, Cone):
            # A truncated cone (frustum) is represented by one quadratic side
            # surface plus two end caps. In local coordinates z is centred at
            # zero and the radius varies linearly:
            #   r(z) = r_mid + slope*z.
            # The side equation is x^2 + y^2 - r(z)^2 = 0. Expanding it gives
            # a quadric, which is then transformed analytically into world
            # coordinates. The cap half-spaces make the cone finite.
            slope = (shape.top_radius - shape.bottom_radius) / shape.height
            r_mid = (shape.bottom_radius + shape.top_radius) / 2
            local_coeff = (1., 1., -slope*slope, 0., 0., 0., 0., 0.,
                           -2*r_mid*slope, -r_mid*r_mid)
            coeff = transform_quadric(local_coeff, rotation, translation)
            side = self.registry.quadric(coeff, f"{name}_cone_side")
            axis = self._unit(rotation @ np.array([0., 0., 1.]))
            upper = self.registry.plane(axis, translation + axis*shape.height/2, f"{name}_cone_top")
            lower = self.registry.plane(axis, translation - axis*shape.height/2, f"{name}_cone_bottom")
            surfaces.extend((side, upper, lower))
            return Intersection((Halfspace(side, -1), Halfspace(upper, -1), Halfspace(lower, 1))), surfaces

        if isinstance(shape, Ellipsoid):
            # Ellipsoid boundary in its local frame:
            #   x^2/rx^2 + y^2/ry^2 + z^2/rz^2 - 1 = 0.
            # A rigid transform preserves the quadric form, so the ellipsoid
            # remains one analytical surface in world coordinates.
            rx, ry, rz = shape.radii
            local_coeff = (1/rx**2, 1/ry**2, 1/rz**2, 0., 0., 0., 0., 0., 0., -1.)
            coeff = transform_quadric(local_coeff, rotation, translation)
            surface = self.registry.quadric(coeff, f"{name}_ellipsoid")
            surfaces.append(surface)
            return Halfspace(surface, -1), surfaces

        if isinstance(shape, Torus):
            # The torus is retained as a dedicated fourth-degree analytical
            # surface, not converted into a quadric or triangle mesh. The axis
            # can be rotated here; an exporter may later restrict which torus
            # axis orientations its target transport code supports.
            local_axis = np.eye(3)[{"x": 0, "y": 1, "z": 2}[shape.axis]]
            axis = self._unit(rotation @ local_axis)
            surface = self.registry.torus(translation, shape.major_radius,
                                          shape.minor_radius, axis, f"{name}_torus")
            surfaces.append(surface)
            return Halfspace(surface, -1), surfaces

        if isinstance(shape, Plane):
            # A plane is described by n.x = d. MiniGeom stores a point as
            # normal*offset in the primitive's local frame, transforms that
            # point and the normal, then asks the registry for the world plane.
            normal = self._unit(rotation @ np.asarray(shape.normal, dtype=float))
            local_point = np.asarray(shape.normal, dtype=float) * shape.offset
            point = translation + rotation @ local_point
            surface = self.registry.plane(normal, point, f"{name}_plane")
            surfaces.append(surface)
            return Halfspace(surface, shape.sense), surfaces

        if isinstance(shape, Quadric):
            # A user-provided quadric is already an implicit surface. Preserve
            # its exact coefficients under the rigid world transform. It may
            # optionally be intersected with six planes supplied by an AABB so
            # a half-space that would otherwise be unbounded becomes finite.
            coeff = transform_quadric(shape.coefficients, rotation, translation)
            surface = self.registry.quadric(coeff, f"{name}_quadric")
            surfaces.append(surface)
            region: Region = Halfspace(surface, shape.sense)
            # A general half-space is not automatically bounded. When the
            # caller explicitly supplies bounds, make them part of the CSG
            # definition instead of treating them as diagnostic metadata only.
            if shape.bounded and shape.bounds is not None:
                lower, upper = (np.asarray(v, dtype=float) for v in shape.bounds)
                bound_terms = []
                for i, local_axis in enumerate(np.eye(3)):
                    axis = self._unit(rotation @ local_axis)
                    upper_surface = self.registry.plane(
                        axis, translation + rotation @ (local_axis * upper[i]),
                        f"{name}_bound_max_{i}")
                    lower_surface = self.registry.plane(
                        axis, translation + rotation @ (local_axis * lower[i]),
                        f"{name}_bound_min_{i}")
                    surfaces.extend((upper_surface, lower_surface))
                    bound_terms.extend((Halfspace(upper_surface, -1),
                                        Halfspace(lower_surface, 1)))
                region = Intersection((region, *bound_terms))
            return region, surfaces

        raise TypeError(f"Unsupported CSG primitive: {type(shape).__name__}")

    def _primitive_region(self, instance: VolumeInstance):
        return self.primitive_region(instance.volume_def.shape, instance.world_matrix,
                                     instance.label.replace("/", "__"))

    @staticmethod
    def _is_void(material: str | None) -> bool:
        return material is None or str(material).strip().lower() in {"", "void", "vacuum", "none"}

    @staticmethod
    def _instance_bounds(instance: VolumeInstance, world_bounds):
        shape = instance.volume_def.shape
        if shape is None:
            return world_bounds
        local = shape.local_bounds()
        if local is None:
            return world_bounds
        lo, hi = (np.asarray(v, dtype=float) for v in local)
        center_local = (lo + hi) / 2
        half_local = (hi - lo) / 2
        r = instance.world_matrix[:3, :3]
        t = instance.world_matrix[:3, 3]
        center = r @ center_local + t
        half = np.abs(r) @ half_local
        return tuple(center-half), tuple(center+half)

    def _append_cell(self, instance, region, *, protruding_root=False, protruding_subtree=False):
        if region is None:
            return
        # Preserve the optional symbolic tag in CSG/JSON. The OpenMC backend
        # decides later whether it is bound to a Material or exported as void.
        material = instance.material
        priority = instance.overlap_priority if protruding_root else instance.local_priority
        self._cells.append(CellSpec(
            id=len(self._cells)+1,
            name=f"cell_{len(self._cells)+1}_{instance.label.replace('/', '__')}",
            path=instance.label, material=material, region=region,
            priority=priority, volume_name=instance.volume_def.name,
            protruding=protruding_subtree,
        ))

    def _append_explicit_region_cell(
        self,
        instance,
        definition,
        region,
        *,
        protruding_root=False,
        protruding_subtree=False,
    ):
        """Append one user-defined Boolean cell to the compiled world model.

        ``definition.region`` has already been transformed into world
        coordinates, clipped to its owner's allowed domain and stripped of
        higher-priority explicit regions. This method only creates the final
        CellSpec record; it does not modify or approximate the region.
        """
        if region is None:
            return
        path = f"{instance.label}/{definition.name}"
        self._cells.append(CellSpec(
            id=len(self._cells) + 1,
            name=f"cell_{len(self._cells)+1}_{path.replace('/', '__')}",
            path=path,
            material=definition.material,
            region=region,
            priority=(instance.overlap_priority if protruding_root else definition.priority),
            volume_name=instance.volume_def.name,
            protruding=protruding_subtree,
        ))

    def _process_node(self, instance, parent_domain, *, protruding_subtree=False,
                      protruding_root=False):
        shape_region = self._shape_regions[instance.path]
        if shape_region is None:
            domain = parent_domain
            if not self._is_void(instance.material):
                raise ValueError(f"Logical volume '{instance.label}' has no primitive but carries material tag '{instance.material}'")
        else:
            domain = shape_region if parent_domain is None else region_intersection(parent_domain, shape_region)
        if domain is None:
            return None

        # Explicit region cells are a second, declarative way to partition one
        # universe. Each expression is defined in the owning VolumeDef's local
        # frame, transformed to world coordinates here, then clipped by the
        # universe's allowed domain (which may itself have been clipped by an
        # ancestor or a protrusion priority rule).
        #
        # To keep cell ownership deterministic, MiniGeom currently requires a
        # universe to use either explicit region cells OR implicit child/lattice
        # placements. Mixing the two would need one shared arbitration list and
        # can otherwise leave gaps or duplicate regions.
        explicit_cells = instance.volume_def.region_cells
        children = [c for c in self._children.get(instance.path, []) if not c.can_overlap]
        if explicit_cells:
            if children or instance.volume_def.lattices:
                raise ValueError(
                    f"Volume '{instance.label}' mixes explicit region cells with "
                    "child placements/lattices. Use one partitioning style per universe."
                )
            if shape_region is None and parent_domain is None:
                raise ValueError(
                    f"Volume '{instance.label}' has explicit cells but no bounded parent domain"
                )

            ordered_cells = sorted(
                explicit_cells,
                key=lambda cell: (cell.priority, cell._order),
                reverse=True,
            )
            occupied_by_higher_priority: list[Region] = []
            for definition in ordered_cells:
                expression_region = definition.region.to_csg(
                    self, parent_matrix=instance.world_matrix,
                    name=f"{instance.label.replace('/', '__')}_{definition.name}",
                )
                # Clip the user expression to the actual universe domain.
                # This guarantees that a cell expression cannot escape its
                # enclosing volume, even when it contains a complement.
                raw_region = region_intersection(domain, expression_region)
                # Preserve the *unresolved* explicit-cell footprint as a raw
                # diagnostic instance. OverlapChecker uses these entries to
                # distinguish geometric intersections from priority-resolved
                # output cells. The owning universe AABB is conservative; it
                # may be larger than the Boolean region, but never excludes it.
                if raw_region is not None:
                    bounds = self._instance_bounds(
                        instance, self._current_world_bounds
                    )
                    self._raw_instances.append(RawInstanceSpec(
                        path=f"{instance.label}/{definition.name}",
                        region=raw_region,
                        bounds=(tuple(bounds[0]), tuple(bounds[1])),
                        material=definition.material,
                        parent_path=instance.label,
                        priority=definition.priority,
                        order=definition._order,
                        protruding=False,
                    ))
                effective_region = region_subtract(
                    raw_region, region_union(occupied_by_higher_priority)
                )
                self._append_explicit_region_cell(
                    instance, definition, effective_region,
                    protruding_root=protruding_root,
                    protruding_subtree=protruding_subtree,
                )
                # Store the raw (not already-subtracted) footprint so every
                # lower-priority cell loses the complete higher-priority set.
                if raw_region is not None:
                    occupied_by_higher_priority.append(raw_region)

            # A tagged enclosing VolumeDef acts as a fallback material for any
            # residual space not claimed by explicit cells. If its material is
            # None, no residual cell is created: the user is declaring the
            # explicit cells to be the intended partition. This avoids emitting
            # a mathematically empty catch-all cell when the explicit regions
            # already tile the complete universe. Any genuine gaps remain
            # visible to find_cells() and to the sampled partition diagnostic.
            own_region = region_subtract(domain, region_union(occupied_by_higher_priority))
            if shape_region is not None and instance.material is not None:
                self._append_cell(instance, own_region, protruding_root=protruding_root,
                                  protruding_subtree=protruding_subtree)
            return domain

        children.sort(key=lambda c: (c.local_priority, c.local_order), reverse=True)
        higher_occupied: list[Region] = []
        all_occupied: list[Region] = []
        for child in children:
            allowed = region_subtract(domain, region_union(higher_occupied))
            occupied = self._process_node(child, allowed,
                                          protruding_subtree=protruding_subtree,
                                          protruding_root=False)
            if occupied is not None:
                higher_occupied.append(occupied)
                all_occupied.append(occupied)

        if shape_region is not None:
            own_region = region_subtract(domain, region_union(all_occupied))
            self._append_cell(instance, own_region, protruding_root=protruding_root,
                              protruding_subtree=protruding_subtree)
            return domain
        return region_union(all_occupied)

    def _root_bounds(self, root: VolumeInstance):
        local = root.volume_def.shape.local_bounds()
        if local is None:
            raise ValueError("Could not determine the root volume bounds")
        lo, hi = (np.asarray(v, dtype=float) for v in local)
        r, t = root.world_matrix[:3, :3], root.world_matrix[:3, 3]
        c, h = r @ ((lo+hi)/2)+t, np.abs(r) @ ((hi-lo)/2)
        return tuple(c-h), tuple(c+h)

    def compile(self, scene) -> CSGModel:
        self.registry = SurfaceRegistry(self.surface_precision)
        self._shape_regions.clear(); self._children.clear(); self._cells.clear(); self._raw_instances.clear()
        instances = list(scene.iter_instances())
        if not instances:
            raise ValueError("The scene contains no instances")
        root = instances[0]
        if root.volume_def.shape is None or not root.volume_def.shape.is_bounded:
            raise ValueError("The root volume must be a bounded solid (Box, Sphere, Cylinder, Cone, Ellipsoid, Torus, or explicitly bounded Quadric)")
        world_bounds = self._root_bounds(root)
        self._current_world_bounds = world_bounds

        for instance in instances:
            region, surfaces = self._primitive_region(instance)
            self._shape_regions[instance.path] = region
            self._children.setdefault(instance.container_path, []).append(instance)
            if instance.path == root.path:
                self.registry.mark_boundary(surfaces, self.boundary_type)
            if region is not None:
                bounds = self._instance_bounds(instance, world_bounds)
                self._raw_instances.append(RawInstanceSpec(
                    path=instance.label, region=region,
                    bounds=(tuple(bounds[0]), tuple(bounds[1])),
                    material=instance.material,
                    parent_path="/".join(instance.container_path) if instance.container_path else None,
                    priority=instance.overlap_priority if instance.can_overlap else instance.local_priority,
                    order=instance.global_order, protruding=instance.can_overlap,
                    lattice_name=instance.lattice_name,
                    lattice_index=instance.lattice_index,
                ))

        world_region = self._shape_regions[root.path]
        self._process_node(root, None)

        protruders = [instance for instance in instances if instance.can_overlap]
        for instance in protruders:
            if self._shape_regions[instance.path] is None:
                raise ValueError(f"Protruding placement '{instance.label}' must have a primitive")
        protruders.sort(key=lambda item: (item.overlap_priority, item.global_order), reverse=True)
        raw_protrusion_union = region_union(self._shape_regions[item.path] for item in protruders)

        # Ordinary cells yield to every protruding footprint.
        if raw_protrusion_union is not None:
            ordinary_count = len(self._cells)
            for index in range(ordinary_count):
                self._cells[index].region = region_subtract(self._cells[index].region, raw_protrusion_union)

        higher_protruding_regions: list[Region] = []
        for instance in protruders:
            allowed = region_subtract(world_region, region_union(higher_protruding_regions))
            group_domain = region_intersection(allowed, self._shape_regions[instance.path])
            self._process_node(instance, group_domain, protruding_subtree=True, protruding_root=True)
            higher_protruding_regions.append(group_domain)

        return CSGModel(scene_name=scene.name, top_volume=root.volume_def.name,
                        surfaces=self.registry.surfaces, cells=self._cells,
                        raw_instances=self._raw_instances, world_bounds=world_bounds)
