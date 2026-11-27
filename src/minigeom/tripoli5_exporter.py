"""TRIPOLI-5 AGORA exporter for MiniGeom analytical scenes.

The exporter maps MiniGeom's analytical Region tree to AGORA Shapes, creates
AGORA Volumes and CompositeVolumes, and preserves compatible rectangular
lattices as native AGORA RectLattice objects. It does not create nuclear
material definitions: MiniGeom material tags are passed to AGORA as strings.

TRIPOLI-5 is an optional dependency. Importing MiniGeom does not require it.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np

from .csg import (
    CSGCompiler, Complement, Halfspace, Intersection, Region, UnionRegion,
    region_intersection, region_subtract, region_union,
)
from .geometry import LatticePlacement, Placement, VolumeDef, _rotation_xyz

try:
    import tripoli5.agora as agora
    from tripoli5.core.euclide import (
        AffineTransformation, Point, Rotation, Translation, Vector,
    )
except ImportError:
    agora = None
    AffineTransformation = Point = Rotation = Translation = Vector = None


class Tripoli5Exporter:
    """Export a MiniGeom scene to TRIPOLI-5's AGORA geometry model.

    ``mode='auto'`` preserves nested universes and native rectangular lattices
    when the scene has shaped nodes and no protruding placements. It falls back
    to flattened CSG for logical shape-less groups and protruding placements.
    ``mode='flat'`` always uses a single AGORA universe. ``mode='hierarchy'``
    requires the native hierarchy representation to be supported.

    TRIPOLI-5 AGORA Volume leaves require a material-name string. Any MiniGeom
    leaf without a symbolic tag must therefore be assigned ``default_material``
    or export fails with a message listing the missing paths.
    """

    def __init__(self, scene, *, boundary_type: str = "leakage",
                 mode: str = "auto", default_material: str | None = None,
                 use_lattices: bool = True, optimize_repeated: bool = True):
        if agora is None:
            raise ImportError(
                "TRIPOLI-5 Python API is not installed. Install the TRIPOLI-5 "
                "Python package to use Tripoli5Exporter."
            )
        if mode not in {"auto", "flat", "hierarchy"}:
            raise ValueError("mode must be 'auto', 'flat' or 'hierarchy'")
        if boundary_type not in {"leakage", "vacuum", "reflective", "reflection"}:
            raise ValueError(
                "boundary_type must be 'leakage'/'vacuum' or 'reflective'/'reflection'"
            )
        if default_material is not None and (not isinstance(default_material, str)
                                              or not default_material.strip()):
            raise ValueError("default_material must be a non-empty string or None")

        self.scene = scene
        self.boundary_type = boundary_type
        self.mode = mode
        self.default_material = default_material.strip() if default_material else None
        self.use_lattices = bool(use_lattices)
        self.optimize_repeated = bool(optimize_repeated)
        self.compiler = CSGCompiler(boundary_type="vacuum")
        self.geometry = None
        self.root_universe = None
        self.universes: dict[str, object] = {}
        self.lattices: dict[str, object] = {}
        self.volumes: list[object] = []
        self.surfaces: dict[int, object] = {}
        self._universe_cache: dict[tuple, object] = {}
        self._selected_mode: str | None = None
        self._fallback_reason: str | None = None
        self._root_boundary_specs = []
        self._missing_material_paths: list[str] = []

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:100] or "unnamed"

    def _hierarchy_incompatibility(self) -> str | None:
        instances = list(self.scene.iter_instances())
        if any(item.can_overlap for item in instances):
            return "protruding placements require the flattened CSG backend"
        for item in instances:
            if item.volume_def.region_cells:
                return (f"volume '{item.label}' contains explicit Boolean region cells; "
                        "the flat CSG backend is required")
            if item.volume_def.shape is None:
                return f"logical node '{item.label}' has no bounding primitive"
        if not self.use_lattices and any(v.lattices for v in self.scene.defs.values()):
            return "use_lattices=False while the scene contains a rectangular lattice"
        return None

    def _select_mode(self) -> str:
        if self.mode != "auto":
            if self.mode == "hierarchy":
                reason = self._hierarchy_incompatibility()
                if reason:
                    raise ValueError(f"Hierarchical export is not possible: {reason}")
            return self.mode
        reason = self._hierarchy_incompatibility()
        if reason:
            self._fallback_reason = reason
            return "flat"
        return "hierarchy"

    def _make_agora_surface(self, spec):
        coeff = spec.coefficients
        label = self._safe_name(spec.name)
        if spec.kind == "sphere":
            x0, y0, z0, radius = coeff
            return agora.Sphere(Point(x0, y0, z0), radius, label=label)
        if spec.kind == "plane":
            a, b, c, d = coeff
            return agora.Plane(a, b, c, d, label=label)
        if spec.kind == "quadric":
            return agora.Quadric(*coeff, label=label)
        if spec.kind == "torus":
            raise NotImplementedError(
                f"TRIPOLI-5 AGORA does not expose a torus surface ({spec.name}). "
                "Replace this primitive with supported analytical surfaces or use STL export."
            )
        raise TypeError(f"Unsupported CSG surface type: {spec.kind}")

    def _surface(self, spec):
        if spec.id not in self.surfaces:
            self.surfaces[spec.id] = self._make_agora_surface(spec)
        return self.surfaces[spec.id]

    def _region(self, region: Region):
        if isinstance(region, Halfspace):
            surface = self._surface(region.surface)
            return -surface if region.sense < 0 else +surface
        if isinstance(region, Intersection):
            result = self._region(region.terms[0])
            for term in region.terms[1:]:
                result = result & self._region(term)
            return result
        if isinstance(region, UnionRegion):
            result = self._region(region.terms[0])
            for term in region.terms[1:]:
                result = result | self._region(term)
            return result
        if isinstance(region, Complement):
            return -self._region(region.term)
        raise TypeError(f"Unsupported CSG region node: {type(region).__name__}")

    def _shape(self, region: Region):
        return agora.Shape(self._region(region))

    def _material_tag(self, tag: str | None, path: str) -> str:
        result = tag.strip() if isinstance(tag, str) else None
        result = result or self.default_material
        if not result:
            self._missing_material_paths.append(path)
            return "__MINIGEOM_MISSING_MATERIAL__"
        return result

    def _affine_transformation(self, position, rotation):
        """Map MiniGeom's ``R @ point + translation`` to AGORA's transform.

        AGORA composes a translation followed by a rotation. Therefore its
        pre-rotation translation must be ``R.T @ translation`` so the final
        transform matches MiniGeom's parent-coordinate convention.
        """
        pos = np.asarray(position, dtype=float)
        rot = np.asarray(rotation, dtype=float)
        rotation_matrix = _rotation_xyz(tuple(float(v) for v in rot))
        pre_rotation_translation = rotation_matrix.T @ pos
        translation = (
            None if np.allclose(pre_rotation_translation, 0.0)
            else Translation(Vector(*map(float, pre_rotation_translation)))
        )
        rotation_obj = None if np.allclose(rot, 0.0) else Rotation(Vector(*map(float, rot)))
        if translation is None and rotation_obj is None:
            return None
        return AffineTransformation(translation, rotation_obj)

    def _box_region(self, lower, upper, name):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        terms = []
        for axis_index, axis in enumerate(np.eye(3)):
            hi = self.compiler.registry.plane(
                axis, axis * upper[axis_index], f"{name}_max_{axis_index}"
            )
            lo = self.compiler.registry.plane(
                axis, axis * lower[axis_index], f"{name}_min_{axis_index}"
            )
            from .csg import Halfspace as CSGHalfspace
            terms.extend((CSGHalfspace(hi, -1), CSGHalfspace(lo, 1)))
        return Intersection(tuple(terms))

    def _build_universe(self, volume: VolumeDef, root_material_override=None):
        cache_key = (id(volume), root_material_override) if self.optimize_repeated else (
            id(volume), root_material_override, len(self._universe_cache)
        )
        if cache_key in self._universe_cache:
            return self._universe_cache[cache_key]

        universe_label = f"u_{self._safe_name(volume.name)}"
        if root_material_override:
            universe_label += f"_{self._safe_name(root_material_override)}"
        domain, root_surfaces = self.compiler.primitive_region(
            volume.shape, np.eye(4), f"{volume.name}_local"
        )
        if domain is None:
            raise ValueError(f"Volume definition '{volume.name}' has no analytical shape")
        if volume is self.scene.top_volume:
            self._root_boundary_specs = list(root_surfaces)

        # Higher-priority children are inserted first. Their raw footprints are
        # removed from lower-priority siblings and from the parent's own region.
        events = [(item._order, "placement", item) for item in volume.children]
        if self.use_lattices:
            events.extend((item._order, "lattice", item) for item in volume.lattices)
        events.sort(key=lambda event: (event[2].priority, event[0]), reverse=True)

        ag_volumes = []
        occupied: list[Region] = []
        for _, kind, item in events:
            if kind == "placement":
                placement: Placement = item
                child = placement.child
                if child.shape is None:
                    raise ValueError(f"Placement '{placement.name}' has no analytical shape")
                child_footprint, _ = self.compiler.primitive_region(
                    child.shape, placement.local_matrix(),
                    f"{volume.name}_{placement.name}_footprint",
                )
                raw_region = region_intersection(domain, child_footprint)
                effective_region = region_subtract(raw_region, region_union(occupied))
                child_universe = self._build_universe(child, placement.material_override)
                composite = agora.CompositeVolume(
                    shape=self._shape(effective_region),
                    filler=child_universe,
                    affinetransformation=self._affine_transformation(
                        placement.position, placement.rotation
                    ),
                    label=self._safe_name(placement.name),
                )
                ag_volumes.append(composite)
                occupied.append(raw_region)
            else:
                lattice_spec: LatticePlacement = item
                nx, ny, nz = lattice_spec.shape
                child_universe = self._build_universe(
                    lattice_spec.child, lattice_spec.material_override
                )
                lattice = agora.RectLattice(
                    point=Point(*lattice_spec.lower_left),
                    pitch=Vector(*lattice_spec.pitch),
                    nx=nx, ny=ny, nz=nz,
                    universe=child_universe,
                    outer_universe=None,
                    label=self._safe_name(lattice_spec.name),
                )
                # Native AGORA lattices support per-element universe
                # substitution. MiniGeom indices are natural (i,j,k) order.
                # We keep the regular lattice intact and replace only the
                # specified entries, avoiding synthetic overlap cells.
                for (i, j, k), replacement_def in lattice_spec.replacements.items():
                    replacement_universe = self._build_universe(
                        replacement_def, lattice_spec.material_override
                    )
                    lattice[i, j, k] = replacement_universe
                self.lattices[lattice_spec.name] = lattice
                envelope = self._box_region(
                    lattice_spec.lower_left, lattice_spec.upper_right,
                    f"{volume.name}_{lattice_spec.name}_envelope",
                )
                raw_region = region_intersection(domain, envelope)
                effective_region = region_subtract(raw_region, region_union(occupied))
                composite = agora.CompositeVolume(
                    shape=self._shape(effective_region),
                    filler=lattice,
                    label=self._safe_name(lattice_spec.name),
                )
                ag_volumes.append(composite)
                occupied.append(raw_region)

        own_region = region_subtract(domain, region_union(occupied))
        own_tag = root_material_override if root_material_override is not None else volume.material
        own_material = self._material_tag(own_tag, volume.name)
        own_volume = agora.Volume(
            shape=self._shape(own_region),
            material=own_material,
            label=f"{self._safe_name(volume.name)}_body",
        )
        ag_volumes.append(own_volume)
        self.volumes.extend(ag_volumes)
        # AGORA Universe is constructed from a sequence of leaf/composite volumes.
        universe = agora.Universe(ag_volumes, label=universe_label)
        self._universe_cache[cache_key] = universe
        self.universes[universe_label] = universe
        return universe

    def _make_boundary_conditions(self):
        conditions = agora.BoundaryConditions()
        if self.boundary_type in {"leakage", "vacuum"}:
            condition = agora.LeakageBoundaryCondition()
        else:
            condition = agora.ReflectionBoundaryCondition()
        for spec in self._root_boundary_specs:
            conditions.add(self._surface(spec), condition)
        return conditions

    def _build_hierarchy(self):
        self.compiler.registry = type(self.compiler.registry)(self.compiler.surface_precision)
        self._universe_cache.clear()
        self.universes.clear()
        self.lattices.clear()
        self.volumes.clear()
        self.surfaces.clear()
        self._root_boundary_specs = []
        self._missing_material_paths = []
        root = self._build_universe(self.scene.top_volume)
        if self._missing_material_paths:
            missing = ", ".join(sorted(set(self._missing_material_paths)))
            raise ValueError(
                "TRIPOLI-5 requires a material-name string for every material leaf. "
                f"Assign material tags or pass default_material. Missing tags at: {missing}"
            )
        self.root_universe = root
        bcs = self._make_boundary_conditions()
        self.geometry = agora.Geometry(
            root_universe=root, boundary_conditions=bcs,
            description=f"MiniGeom scene: {self.scene.name}",
        )
        return self.geometry

    def _build_flat(self):
        self.compiler = CSGCompiler(boundary_type="vacuum")
        model = self.compiler.compile(self.scene)
        self.universes.clear()
        self.lattices.clear()
        self.volumes.clear()
        self.surfaces.clear()
        self._root_boundary_specs = [spec for spec in model.surfaces
                                     if spec.boundary_type not in {"transmission", "none"}]
        ag_volumes = []
        self._missing_material_paths = []
        for cell in model.cells:
            material = self._material_tag(cell.material, cell.path)
            ag_volumes.append(agora.Volume(
                shape=self._shape(cell.region), material=material,
                label=self._safe_name(cell.path),
            ))
        if self._missing_material_paths:
            missing = ", ".join(sorted(set(self._missing_material_paths)))
            raise ValueError(
                "TRIPOLI-5 requires a material-name string for every material leaf. "
                f"Assign material tags or pass default_material. Missing tags at: {missing}"
            )
        root = agora.Universe(ag_volumes, label=f"{self._safe_name(model.scene_name)}_root")
        self.volumes.extend(ag_volumes)
        self.root_universe = root
        bcs = self._make_boundary_conditions()
        self.geometry = agora.Geometry(
            root_universe=root, boundary_conditions=bcs,
            description=f"MiniGeom scene: {model.scene_name}",
        )
        self.universes = {root.label: root}
        return self.geometry

    def build(self):
        """Build and return a ``tripoli5.agora.Geometry`` instance."""
        self._selected_mode = self._select_mode()
        if self._selected_mode == "hierarchy":
            return self._build_hierarchy()
        return self._build_flat()

    def export(self, filename: str | Path = "geometry.agora", *,
               manifest: bool = True) -> Path:
        """Build and serialize an AGORA binary geometry file."""
        geometry = self.build()
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        geometry.serialize(filename=str(path))
        if manifest:
            manifest_data = {
                "scene": self.scene.name,
                "format": "TRIPOLI-5 AGORA",
                "geometry_file": path.name,
                "export_mode": self._selected_mode,
                "fallback_reason": self._fallback_reason,
                "universes": len(self.universes),
                "lattices": len(self.lattices),
                "volumes": len(self.volumes),
                "material_tags": sorted({
                    str(instance.material).strip()
                    for instance in self.scene.iter_instances()
                    if isinstance(instance.material, str) and instance.material.strip()
                }),
                "boundary_type": self.boundary_type,
                "note": (
                    "TRIPOLI-5 AGORA stores material labels as strings. Material "
                    "compositions and nuclear data are defined separately by TRIPOLI-5."
                ),
            }
            path.with_suffix(path.suffix + ".json").write_text(
                json.dumps(manifest_data, indent=2), encoding="utf-8"
            )
        return path
