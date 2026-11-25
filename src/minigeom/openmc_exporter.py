"""OpenMC exporter: flat CSG cells or optimized hierarchical universes/lattices."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import re
import xml.etree.ElementTree as ET

import numpy as np

from .csg import (
    CSGCompiler, CSGModel, Complement, Halfspace, Intersection, UnionRegion,
    Region, SurfaceRegistry, region_intersection, region_subtract, region_union,
)
from .geometry import LatticePlacement, Placement, VolumeDef

try:
    import openmc
except ImportError:
    openmc = None


class OpenMCExporter:
    """Export a MiniGeom scene to OpenMC XML.

    ``mode='auto'`` uses shared universes and native RectLattice whenever the
    hierarchy is compatible (shaped definitions and no protruding placements).
    It falls back to a flattened world-coordinate CSG model for shape-less
    logical groups or protruding placements. ``mode='flat'`` always flattens;
    ``mode='hierarchy'`` requires the optimized hierarchy representation.
    """

    def __init__(self, scene, materials_map: Optional[dict] = None,
                 boundary_type: str = "vacuum", *, mode: str = "auto",
                 optimize_repeated: bool = True, use_lattices: bool = True,
                 material_ids: Optional[dict[str, int]] = None):
        if openmc is None:
            raise ImportError("OpenMC is not installed. Install OpenMC to export XML files.")
        if mode not in {"auto", "flat", "hierarchy"}:
            raise ValueError("mode must be 'auto', 'flat' or 'hierarchy'")
        self.scene = scene
        # Symbolic tags are always exported as OpenMC material IDs. Real
        # Material objects are optional and only control materials.xml output.
        self.materials_map = self._normalize_mapping(materials_map or {}, "materials_map")
        self.requested_material_ids = self._normalize_mapping(material_ids or {}, "material_ids")
        self.material_ids: dict[str, int] = {}
        self.boundary_type = boundary_type
        self.mode = mode
        self.optimize_repeated = bool(optimize_repeated)
        self.use_lattices = bool(use_lattices)
        self.compiler = CSGCompiler(boundary_type=boundary_type)
        self.model: CSGModel | None = None
        self.geometry = None
        self.materials = None
        self.surfaces = []
        self.cells = []
        self.universes = {}
        self.lattices = {}
        self._surface_objects = {}
        self._used_material_names: set[str] = set()
        self._unbound_material_names: set[str] = set()
        # Cells without an in-memory Material object are exported as void by
        # OpenMC's Python model, then their XML material attribute is patched to
        # the symbolic material ID. This avoids creating fake Material objects
        # and avoids modifying OpenMC's process-wide material ID registry.
        self._pending_cell_material_ids: dict[int, int] = {}
        self._universe_cache = {}
        self._selected_mode: str | None = None
        self._fallback_reason: str | None = None

    @staticmethod
    def _normalize_mapping(mapping: dict, label: str) -> dict:
        normalized = {}
        for key, value in mapping.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} keys must be non-empty strings")
            tag = key.strip()
            if tag in normalized:
                raise ValueError(f"Duplicate {label} key after whitespace normalization: {tag!r}")
            normalized[tag] = value
        return normalized

    @staticmethod
    def _is_void(name: str | None) -> bool:
        return name is None or str(name).strip().lower() in {"", "void", "vacuum", "none"}

    @staticmethod
    def _safe_name(name: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        return value[:100] or "unnamed"

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

    def _hierarchy_incompatibility(self) -> str | None:
        instances = list(self.scene.iter_instances())
        if any(inst.can_overlap for inst in instances):
            return "protruding placements currently require the flattened CSG backend"
        for inst in instances:
            if inst.volume_def.shape is None:
                return f"logical node '{inst.label}' must be flattened"
        if not self.use_lattices and any(v.lattices for v in self.scene.defs.values()):
            return "use_lattices=False while the scene contains a RectLattice"
        return None

    def compile_csg(self) -> CSGModel:
        self.model = self.compiler.compile(self.scene)
        return self.model

    def _make_openmc_surface(self, spec):
        common = {"name": spec.name, "boundary_type": spec.boundary_type}
        coeff = spec.coefficients
        if spec.kind == "sphere":
            x0, y0, z0, radius = coeff
            return openmc.Sphere(x0=x0, y0=y0, z0=z0, r=radius, **common)
        if spec.kind == "plane":
            a, b, c, d = coeff
            return openmc.Plane(a=a, b=b, c=c, d=d, **common)
        if spec.kind == "quadric":
            return openmc.Quadric(*coeff, **common)
        if spec.kind == "torus":
            x0, y0, z0, major, minor, ux, uy, uz = coeff
            axis = np.asarray([ux, uy, uz], dtype=float)
            axes = np.eye(3)
            match = next((idx for idx, basis in enumerate(axes)
                          if abs(abs(float(np.dot(axis, basis))) - 1.0) < 1e-8), None)
            if match is None:
                raise ValueError(
                    f"Torus '{spec.name}' has an axis that is not parallel to X, Y or Z. "
                    "OpenMC only provides XTorus, YTorus and ZTorus; align the torus axis."
                )
            cls = (openmc.XTorus, openmc.YTorus, openmc.ZTorus)[match]
            return cls(x0=x0, y0=y0, z0=z0, a=major, b=minor, c=minor, **common)
        raise TypeError(f"Unsupported CSG surface type: {spec.kind}")

    def _region_to_openmc(self, region: Region):
        if isinstance(region, Halfspace):
            surface_id = region.surface.id
            if surface_id not in self._surface_objects:
                self._surface_objects[surface_id] = self._make_openmc_surface(region.surface)
            surface = self._surface_objects[surface_id]
            return -surface if region.sense < 0 else +surface
        if isinstance(region, Intersection):
            result = self._region_to_openmc(region.terms[0])
            for term in region.terms[1:]:
                result = result & self._region_to_openmc(term)
            return result
        if isinstance(region, UnionRegion):
            result = self._region_to_openmc(region.terms[0])
            for term in region.terms[1:]:
                result = result | self._region_to_openmc(term)
            return result
        if isinstance(region, Complement):
            return ~self._region_to_openmc(region.term)
        raise TypeError(f"Unknown CSG region node: {type(region).__name__}")

    def _prepare_material_catalog(self):
        """Assign deterministic OpenMC material IDs to non-void symbolic tags."""
        if self.model is None:
            raise RuntimeError("Compile the scene before preparing material IDs")

        tags = sorted({str(cell.material).strip() for cell in self.model.cells
                       if not self._is_void(cell.material)})
        unused_explicit = set(self.requested_material_ids) - set(tags)
        if unused_explicit:
            raise ValueError(
                "material_ids contains tags not used by the compiled geometry: "
                + ", ".join(sorted(unused_explicit))
            )

        ids: dict[str, int] = {}
        id_owners: dict[int, tuple[str, object | None]] = {}

        for tag, value in self.requested_material_ids.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Material ID for '{tag}' must be a positive integer")
            previous = id_owners.get(value)
            if previous is not None and previous[0] != tag:
                raise ValueError(
                    f"Material ID {value} is assigned to both '{previous[0]}' and '{tag}'"
                )
            ids[tag] = value
            id_owners[value] = (tag, None)

        # Material objects bind only the tags actually present in this scene.
        for tag, material in self.materials_map.items():
            if not isinstance(material, openmc.Material):
                raise TypeError(f"materials_map[{tag!r}] must be an openmc.Material")
            if tag not in tags:
                continue
            material_id = int(material.id)
            if tag in ids and ids[tag] != material_id:
                raise ValueError(
                    f"Material ID mismatch for '{tag}': requested {ids[tag]}, "
                    f"but the supplied openmc.Material has ID {material_id}"
                )
            previous = id_owners.get(material_id)
            if previous is not None and previous[0] != tag:
                same_object = previous[1] is material
                if not same_object:
                    raise ValueError(
                        f"Material ID {material_id} is assigned to both "
                        f"'{previous[0]}' and '{tag}'"
                    )
            ids[tag] = material_id
            id_owners[material_id] = (tag, material)

        reserved_ids = set(ids.values())
        next_id = 1
        for tag in tags:
            if tag in ids:
                continue
            while next_id in reserved_ids:
                next_id += 1
            ids[tag] = next_id
            reserved_ids.add(next_id)
            next_id += 1

        # Auto IDs are intentionally deterministic for a given set of tags.
        # Callers that merge several geometry decks should provide material_ids.
        self.material_ids = {tag: ids[tag] for tag in tags}
        self._used_material_names = set()
        self._unbound_material_names = {tag for tag in tags if tag not in self.materials_map}
        self._pending_cell_material_ids = {}

    def _collect_materials(self):
        """Collect only real Material objects referenced by exported cells."""
        used, seen = [], set()
        for name in sorted(self._used_material_names):
            material = self.materials_map[name]
            if material.id not in seen:
                used.append(material)
                seen.add(material.id)
        self.materials = openmc.Materials(used) if used else None

    def _material(self, name):
        """Return an actual OpenMC Material, or None for an external symbolic tag."""
        if self._is_void(name):
            return None
        tag = str(name).strip()
        if tag not in self.material_ids:
            raise RuntimeError(
                f"Material tag '{tag}' was not registered. Compile the CSG model first."
            )
        material = self.materials_map.get(tag)
        if material is None:
            self._unbound_material_names.add(tag)
            return None
        if not isinstance(material, openmc.Material):
            raise TypeError(f"materials_map[{tag!r}] must be an openmc.Material")
        self._used_material_names.add(tag)
        return material

    def _record_external_material(self, cell, material_tag):
        """Remember the numeric material reference for a symbolic-only cell."""
        if self._is_void(material_tag):
            return
        tag = str(material_tag).strip()
        if tag not in self.materials_map:
            self._pending_cell_material_ids[int(cell.id)] = self.material_ids[tag]

    @staticmethod
    def _patch_geometry_material_ids(filename: str | Path, pending: dict[int, int]):
        """Patch symbolic material IDs into OpenMC XML without fake Material objects."""
        if not pending:
            return
        path = Path(filename)
        tree = ET.parse(path)
        root = tree.getroot()
        found = set()
        for element in root.iter("cell"):
            cell_id_text = element.get("id")
            if cell_id_text is None:
                continue
            cell_id = int(cell_id_text)
            if cell_id not in pending:
                continue
            current = element.get("material")
            if current not in (None, "void", str(pending[cell_id])):
                raise ValueError(
                    f"Cannot replace material reference on OpenMC cell {cell_id}: "
                    f"XML currently contains {current!r}"
                )
            element.set("material", str(pending[cell_id]))
            found.add(cell_id)
        missing = set(pending) - found
        if missing:
            raise RuntimeError(
                "Could not find OpenMC cell IDs in exported geometry.xml: "
                + ", ".join(str(value) for value in sorted(missing))
            )
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def _prepare_surface_objects(self):
        # Create only missing surfaces: hierarchy cells may already reference
        # lazily created OpenMC surface objects from this same registry.
        for spec in self.compiler.registry.surfaces:
            if spec.id not in self._surface_objects:
                self._surface_objects[spec.id] = self._make_openmc_surface(spec)
        self.surfaces = list(self._surface_objects.values())

    def _build_flat(self):
        model = self.compile_csg()
        self._prepare_material_catalog()
        self.compiler.registry  # registry was populated by compile()
        self._surface_objects = {spec.id: self._make_openmc_surface(spec) for spec in model.surfaces}
        self.surfaces = list(self._surface_objects.values())
        self._used_material_names = set()
        self._unbound_material_names = set()
        self._pending_cell_material_ids = {}
        self.cells = []
        for spec in model.cells:
            fill = self._material(spec.material)
            cell = openmc.Cell(name=spec.name, fill=fill,
                               region=self._region_to_openmc(spec.region))
            self._record_external_material(cell, spec.material)
            self.cells.append(cell)
        self._collect_materials()
        root = openmc.Universe(name=f"{model.scene_name}_root")
        root.add_cells(self.cells)
        self.geometry = openmc.Geometry(root, merge_surfaces=False)
        self.universes = {root.name: root}
        self.lattices = {}
        return self.geometry

    def _box_region(self, lower, upper, name):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        terms = []
        for i, axis in enumerate(np.eye(3)):
            hi = self.compiler.registry.plane(axis, np.eye(3)[i]*upper[i], f"{name}_max_{i}")
            lo = self.compiler.registry.plane(axis, np.eye(3)[i]*lower[i], f"{name}_min_{i}")
            terms.extend((Halfspace(hi, -1), Halfspace(lo, 1)))
        return Intersection(tuple(terms))

    def _hierarchy_items(self, volume: VolumeDef):
        items = [(item._order, "placement", item) for item in volume.children]
        if self.use_lattices:
            items += [(item._order, "lattice", item) for item in volume.lattices]
        return sorted(items, key=lambda x: (x[2].priority, x[0]), reverse=True)

    def _build_universe_hierarchy(self, volume: VolumeDef, root_material_override=None):
        key = (id(volume), root_material_override) if self.optimize_repeated else (id(volume), root_material_override, len(self._universe_cache))
        if key in self._universe_cache:
            return self._universe_cache[key]

        name = f"u_{self._safe_name(volume.name)}"
        if root_material_override is not None:
            name += f"_{self._safe_name(str(root_material_override))}"
        universe = openmc.Universe(name=name)
        # Cache before descending; cyclic definitions are rejected by VolumeDef.place.
        self._universe_cache[key] = universe
        self.universes[name] = universe

        local_matrix = np.eye(4)
        domain, root_surfaces = self.compiler.primitive_region(volume.shape, local_matrix,
                                                               f"{volume.name}_local")
        if domain is None:
            raise ValueError(f"Volume definition '{volume.name}' has no primitive")
        if volume is self.scene.top_volume:
            self.compiler.registry.mark_boundary(root_surfaces, self.boundary_type)

        footprints = []
        cells = []
        for order, kind, item in self._hierarchy_items(volume):
            if kind == "placement":
                placement: Placement = item
                if placement.can_overlap:
                    raise ValueError("Hierarchical mode does not support place_protruding")
                if placement.child.shape is None:
                    raise ValueError(f"Logical placement '{placement.name}' must be flattened")
                footprint, _ = self.compiler.primitive_region(
                    placement.child.shape, placement.local_matrix(),
                    f"{volume.name}_{placement.name}_footprint")
                raw_occupied = region_intersection(domain, footprint)
                region = region_subtract(raw_occupied, region_union(footprints))
                child_universe = self._build_universe_hierarchy(
                    placement.child, placement.material_override)
                child_cell = openmc.Cell(
                    name=f"{self._safe_name(volume.name)}__{self._safe_name(placement.name)}",
                    fill=child_universe, region=self._region_to_openmc(region))
                child_cell.translation = placement.position
                if not np.allclose(placement.rotation, (0., 0., 0.), atol=1e-12):
                    child_cell.rotation = placement.rotation
                cells.append(child_cell)
                self.cells.append(child_cell)
                footprints.append(raw_occupied)
            else:
                lattice_spec: LatticePlacement = item
                nx, ny, nz = lattice_spec.shape
                lo = np.asarray(lattice_spec.lower_left, dtype=float)
                hi = np.asarray(lattice_spec.upper_right, dtype=float)
                footprint = self._box_region(lo, hi, f"{volume.name}_{lattice_spec.name}_envelope")
                raw_occupied = region_intersection(domain, footprint)
                region = region_subtract(raw_occupied, region_union(footprints))

                element_universe = self._build_universe_hierarchy(
                    lattice_spec.child, lattice_spec.material_override)
                lattice = openmc.RectLattice(name=self._safe_name(lattice_spec.name))
                lattice.lower_left = lattice_spec.lower_left
                lattice.pitch = lattice_spec.pitch
                # RectLattice.universes uses [z, y-from-high-to-low, x].
                array = np.empty((nz, ny, nx), dtype=object)
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            array[k, ny - 1 - j, i] = element_universe
                lattice.universes = array
                # A finite parent cell clips the lattice envelope. The outer
                # universe is nevertheless supplied to satisfy OpenMC's full
                # RectLattice definition; it is only reached outside the array.
                outer = openmc.Universe(name=f"{lattice.name}_outer")
                outer_cell = openmc.Cell(name=f"{lattice.name}_outer_void")
                outer.add_cell(outer_cell)
                lattice.outer = outer
                self.universes[outer.name] = outer
                self.cells.append(outer_cell)
                self.lattices[lattice.name] = lattice
                lattice_cell = openmc.Cell(
                    name=f"{self._safe_name(volume.name)}__{self._safe_name(lattice_spec.name)}",
                    fill=lattice, region=self._region_to_openmc(region))
                cells.append(lattice_cell)
                self.cells.append(lattice_cell)
                footprints.append(raw_occupied)

        own_material = root_material_override if root_material_override is not None else volume.material
        own_region = region_subtract(domain, region_union(footprints))
        own_cell = openmc.Cell(
            name=f"{self._safe_name(volume.name)}__body",
            fill=self._material(own_material),
            region=self._region_to_openmc(own_region))
        self._record_external_material(own_cell, own_material)
        cells.append(own_cell)
        self.cells.append(own_cell)
        universe.add_cells(cells)
        return universe

    def _build_hierarchy(self):
        # Reuse CSGCompiler's analytical surface/region factory, but build local
        # universe regions so identical VolumeDefs only need one local surface set.
        self.model = self.compile_csg()  # complete flattened JSON/audit representation
        self._prepare_material_catalog()
        self.compiler.registry = SurfaceRegistry(self.compiler.surface_precision)
        self.compiler._shape_regions.clear()
        self.compiler._children.clear()
        self.compiler._cells.clear()
        self._surface_objects = {}
        self._used_material_names = set()
        self._unbound_material_names = set()
        self._pending_cell_material_ids = {}
        self._universe_cache = {}
        self.universes, self.lattices, self.cells = {}, {}, []

        root_universe = self._build_universe_hierarchy(self.scene.top_volume)
        self._prepare_surface_objects()
        # Cells already hold the correct OpenMC surface instances returned by
        # _region_to_openmc; registry is local-coordinate and can now be exported.
        self._collect_materials()
        self.geometry = openmc.Geometry(root_universe, merge_surfaces=False)
        return self.geometry

    def build(self):
        """Build and return ``openmc.Geometry``."""
        self._selected_mode = self._select_mode()
        if self._selected_mode == "hierarchy":
            return self._build_hierarchy()
        return self._build_flat()

    def export(self, output_dir: str | Path = "openmc_out", *,
               geometry_filename: str = "geometry.xml",
               materials_filename: str = "materials.xml",
               json_filename: str = "minigeom_csg.json",
               material_ids_filename: str = "materials_ids.json",
               export_materials: bool = True) -> Path:
        """Export geometry.xml, MiniGeom CSG JSON and optionally materials.xml.

        Every non-void symbolic tag receives a deterministic OpenMC material ID.
        If a real ``openmc.Material`` is supplied, its own ID and definition are
        used. Otherwise the cell is built as void in memory and the exported XML
        is patched with the external numeric ID; no fake Material object is made.
        ``materials.xml`` is written only when real Material objects are used.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        geometry = self.build()
        geometry_path = out / geometry_filename
        geometry.export_to_xml(path=str(geometry_path))
        self._patch_geometry_material_ids(geometry_path, self._pending_cell_material_ids)

        materials_written = False
        if export_materials and self.materials is not None and len(self.materials) > 0:
            self.materials.export_to_xml(path=str(out / materials_filename))
            materials_written = True

        assert self.model is not None
        self.model.export_json(out / json_filename)
        material_id_payload = {
            "schema_version": 1,
            "scene": self.model.scene_name,
            "materials": {
                tag: {
                    "id": material_id,
                    "source": "openmc.Material" if tag in self.materials_map else "external",
                }
                for tag, material_id in self.material_ids.items()
            },
        }
        (out / material_ids_filename).write_text(
            json.dumps(material_id_payload, indent=2), encoding="utf-8"
        )
        manifest = {
            "scene": self.model.scene_name,
            "export_mode": self._selected_mode,
            "hierarchy_fallback_reason": self._fallback_reason,
            "geometry_xml": geometry_filename,
            "materials_xml": materials_filename if materials_written else None,
            "materials_ids_json": material_ids_filename,
            "materials_exported": materials_written,
            "material_mode": ("mixed" if self._used_material_names and self._unbound_material_names
                              else "bound" if self._used_material_names
                              else "symbolic" if self.material_ids else "none"),
            "unbound_material_tags": sorted(self._unbound_material_names),
            "material_ids": self.material_ids,
            "csg_json": json_filename,
            "flattened_csg_cells": len(self.model.cells),
            "flattened_csg_surfaces": len(self.model.surfaces),
            "openmc_cells": len(self.cells),
            "openmc_universes": len(self.universes),
            "openmc_lattices": len(self.lattices),
            "boundary_type": self.boundary_type,
            "note": (
                "The CSG JSON is the flattened MiniGeom representation. The OpenMC "
                "geometry may use nested universes and lattices. Symbolic tags are "
                "always converted to material IDs; materials.xml is written only "
                "for real openmc.Material objects supplied by the caller."
            ),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def export_geometry(self, output_dir: str | Path = "openmc_geometry", **kwargs) -> Path:
        """Export geometry.xml and MiniGeom JSON without writing materials.xml.

        Symbolic tags are exported as material IDs. Any referenced real
        material definitions remain the caller's responsibility.
        """
        kwargs.pop("export_materials", None)
        return self.export(output_dir, export_materials=False, **kwargs)

    def export_xml(self, output_dir: str | Path = "openmc_out", export_materials: bool = True) -> Path:
        return self.export(output_dir, export_materials=export_materials)

    def export_summary_json(self, path: str | Path) -> None:
        if self.model is None:
            self.compile_csg()
        assert self.model is not None
        self.model.export_json(path)


OpenMCHierarchyExporter = OpenMCExporter
