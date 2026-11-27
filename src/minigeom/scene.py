"""Scene graph, instance traversal and optional mesh visualization/export.

Analytical CSG resolution lives in ``minigeom.csg``. Mesh boolean operations
here are explicitly approximate and are only intended for previews/STL output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .geometry import MeshPrimitive, Placement, VolumeDef, VolumeInstance


def _require_trimesh():
    """Load the optional mesh backend only when a mesh operation is requested."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "This operation requires Trimesh. Install it with "
            "`pip install minigeom-analytic[mesh]`."
        ) from exc
    return trimesh


def _tri_to_pv(mesh: trimesh.Trimesh):
    import pyvista as pv
    pv_mesh = pv.wrap(mesh)
    if not isinstance(pv_mesh, pv.PolyData):
        pv_mesh = pv_mesh.extract_surface()
    return pv_mesh.triangulate()


def _mesh_boolean(operation: str, meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    trimesh = _require_trimesh()
    meshes = [m for m in meshes if m is not None and not m.is_empty]
    if not meshes:
        return None
    try:
        if operation == "union":
            result = meshes[0] if len(meshes) == 1 else trimesh.boolean.union(meshes, engine="manifold")
        elif operation == "intersection":
            result = meshes[0] if len(meshes) == 1 else trimesh.boolean.intersection(meshes, engine="manifold")
        elif operation == "difference":
            result = meshes[0] if len(meshes) == 1 else trimesh.boolean.difference(meshes, engine="manifold")
        else:
            raise ValueError(f"Unsupported mesh boolean operation: {operation}")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Mesh resolution requires manifold3d. Install it with "
            "`pip install manifold3d`. Analytical OpenMC export does not depend on it."
        ) from exc
    if result is None or result.is_empty:
        return None
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))
    return result


def _union_optional(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    return _mesh_boolean("union", meshes) if meshes else None


def _subtract(base: trimesh.Trimesh | None, cutter: trimesh.Trimesh | None):
    if base is None or base.is_empty:
        return None
    if cutter is None or cutter.is_empty:
        return base
    return _mesh_boolean("difference", [base, cutter])


class Scene:
    """Manage volume definitions and their instances under a root volume."""

    def __init__(self, name: str = "scene", top_volume: Optional[VolumeDef] = None):
        self.name = name
        self.defs: dict[str, VolumeDef] = {}
        self.top_volume: Optional[VolumeDef] = None
        if top_volume is not None:
            self.set_top(top_volume)

    def define(self, name: str, shape=None, material: str | None = None,
               self_priority: int = -100) -> VolumeDef:
        volume = VolumeDef(name, shape, material, self_priority)
        self.register(volume)
        return volume

    def register(self, volume_def: VolumeDef, include_children: bool = True) -> VolumeDef:
        visited: set[int] = set()

        def add(current: VolumeDef):
            if id(current) in visited:
                return
            visited.add(id(current))
            if current.name in self.defs and self.defs[current.name] is not current:
                raise ValueError(f"A volume definition named '{current.name}' already exists")
            self.defs[current.name] = current
            if include_children:
                for placement in current.children:
                    add(placement.child)
                for lattice in current.lattices:
                    add(lattice.child)
                    # Sparse substitutions are real reusable universes too, so
                    # register their definitions for exporters and scene lookup.
                    for replacement in lattice.replacements.values():
                        add(replacement)

        add(volume_def)
        if self.top_volume is None:
            self.top_volume = volume_def
        return volume_def

    def set_top(self, top: VolumeDef | str) -> VolumeDef:
        volume = self.defs[top] if isinstance(top, str) else top
        self.register(volume, include_children=True)
        self.top_volume = volume
        return volume

    def import_stl(self, name: str, filename: str | Path, *,
                   material: str | None = None, scale: float = 1.0,
                   process: bool = True, require_watertight: bool = True) -> VolumeDef:
        """Import an STL mesh as a mesh-backed VolumeDef.

        STL files do not store reliable physical units or material composition.
        ``scale`` converts file coordinates to the scene's length unit. The
        resulting MeshPrimitive is intended for visualization/STL workflows;
        analytical OpenMC and TRIPOLI-5 exporters reject it explicitly.
        """
        shape = MeshPrimitive.from_file(
            filename, scale=scale, process=process,
            require_watertight=require_watertight,
        )
        volume = VolumeDef(name=name, shape=shape, material=material)
        self.register(volume, include_children=False)
        return volume

    load_stl = import_stl

    def iter_instances(self, root: Optional[VolumeDef] = None) -> Iterator[VolumeInstance]:
        """Yield every placement occurrence, expanding rectangular lattice entries.

        Lattice instances are expanded for the analytical CSG model and preview.
        The OpenMC hierarchy exporter consumes the original LatticePlacement and
        emits one native RectLattice instead of one cell per element.
        """
        top = root or self.top_volume
        if top is None:
            return
        self.register(top, include_children=True)
        order_counter = 0
        yield VolumeInstance(
            path=(top.name,), volume_def=top, material=top.material,
            world_matrix=np.eye(4, dtype=float), container_path=(),
            local_priority=top.self_priority, local_order=-1,
            global_order=order_counter,
        )
        order_counter += 1

        def walk(parent: VolumeDef, parent_matrix: np.ndarray,
                 parent_path: tuple[str, ...], definition_stack: tuple[int, ...]):
            nonlocal order_counter
            if id(parent) in definition_stack:
                raise ValueError(f"Definition cycle detected below '{parent.name}'")
            stack = definition_stack + (id(parent),)

            # Placements and lattice groups share the same insertion-order counter.
            events = [(item._order, "placement", item) for item in parent.children]
            events += [(item._order, "lattice", item) for item in parent.lattices]
            for _, kind, item in sorted(events, key=lambda event: event[0]):
                if kind == "placement":
                    placements = [(item, None, None)]
                else:
                    nx, ny, nz = item.shape
                    placements = []
                    for k in range(nz):
                        for j in range(ny):
                            for i in range(nx):
                                element_name = f"{item.name}[{i},{j},{k}]"
                                element_child = item.get_element(i, j, k)
                                synthetic = Placement(
                                    child=element_child, copy_no=(k*ny+j)*nx+i+1,
                                    name=element_name, position=item.center(i,j,k),
                                    rotation=(0.,0.,0.), priority=item.priority,
                                    material_override=item.material_override,
                                    _order=item._order,
                                )
                                placements.append((synthetic, item.name, (i,j,k)))

                for placement, lattice_name, lattice_index in placements:
                    child_path = parent_path + (placement.name,)
                    child_matrix = parent_matrix @ placement.local_matrix()
                    material = (placement.material_override
                                if placement.material_override is not None
                                else placement.child.material)
                    inst = VolumeInstance(
                        path=child_path, volume_def=placement.child, material=material,
                        world_matrix=child_matrix, container_path=parent_path,
                        local_priority=placement.priority, local_order=placement._order,
                        global_order=order_counter, can_overlap=placement.can_overlap,
                        overlap_priority=placement.overlap_priority, placement=placement,
                        lattice_name=lattice_name, lattice_index=lattice_index,
                    )
                    order_counter += 1
                    yield inst
                    yield from walk(placement.child, child_matrix, child_path, stack)

        yield from walk(top, np.eye(4, dtype=float), (top.name,), ())

    def check_overlaps(self, *, samples: int = 12000, seed: int = 1729):
        """Sample raw primitive intersections and verify the resolved CSG partition.

        This is a diagnostic, not a proof of non-overlap: small intersections can
        be missed by finite sampling. The returned report distinguishes raw
        geometric overlaps from unresolved multiple-cell assignments.
        """
        from .overlap import OverlapChecker
        return OverlapChecker(samples=samples, seed=seed).check(self)

    def _mesh_resolved_instances(self) -> dict[str, trimesh.Trimesh | None]:
        """Approximate priority-resolved meshes for preview only (Manifold backend)."""
        _require_trimesh()
        instances = list(self.iter_instances())
        if not instances:
            return {}
        by_parent: dict[tuple[str, ...], list[VolumeInstance]] = {}
        for inst in instances[1:]:
            by_parent.setdefault(inst.container_path, []).append(inst)
        raw = {inst.path: inst.world_mesh() for inst in instances}
        unmeshable = [inst.label for inst in instances
                      if inst.volume_def.shape is not None and raw[inst.path] is None]
        if unmeshable:
            raise NotImplementedError(
                "Mesh resolution does not support primitives without a display mesh "
                f"(Plane/Quadric or similar): {', '.join(unmeshable)}. "
                "Analytical CSG export to OpenMC remains available."
            )
        protruders = [i for i in instances if i.can_overlap]
        if any(raw[p.path] is None for p in protruders):
            raise ValueError("A protruding placement must have a geometric primitive")
        protruders_sorted = sorted(protruders,
                                   key=lambda i: (i.overlap_priority, i.global_order), reverse=True)
        protruder_union = _union_optional([raw[p.path] for p in protruders])
        cells: dict[str, trimesh.Trimesh | None] = {}

        def process(inst: VolumeInstance, parent_domain: trimesh.Trimesh | None,
                    *, root_call=False):
            shape_mesh = raw[inst.path]
            if shape_mesh is None:
                domain = parent_domain
            elif parent_domain is None:
                domain = shape_mesh
            else:
                domain = _mesh_boolean("intersection", [parent_domain, shape_mesh])
            if domain is None:
                return None
            children = [c for c in by_parent.get(inst.path, []) if not c.can_overlap]
            children.sort(key=lambda c: (c.local_priority, c.local_order), reverse=True)
            higher_occupied: list[trimesh.Trimesh] = []
            all_occupied: list[trimesh.Trimesh] = []
            for child in children:
                higher_union = _union_optional(higher_occupied)
                allowed = _subtract(domain, higher_union)
                child_occupied = process(child, allowed)
                if child_occupied is not None:
                    higher_occupied.append(child_occupied)
                    all_occupied.append(child_occupied)
            if shape_mesh is not None:
                own_mesh = _subtract(domain, _union_optional(all_occupied))
                cells[inst.label] = own_mesh
                return domain
            return _union_optional(all_occupied)

        root = instances[0]
        if raw[root.path] is None:
            raise ValueError("The root volume must have a primitive for mesh resolution")
        process(root, None)

        # Ordinary geometry is carved by every protruding placement.
        for inst in instances:
            if inst.can_overlap:
                cells.pop(inst.label, None)
        for label, mesh in list(cells.items()):
            cells[label] = _subtract(mesh, protruder_union)

        # Each protruding subtree is independent of its parent, but clipped to
        # the world and to all higher-priority protruding placements.
        world_mesh = raw[root.path]
        for index, protruder in enumerate(protruders_sorted):
            higher = [raw[p.path] for p in protruders_sorted[:index]]
            group_domain = _subtract(raw[protruder.path], _union_optional(higher))
            group_domain = _mesh_boolean("intersection", [group_domain, world_mesh]) if group_domain else None
            if group_domain is not None:
                process(protruder, group_domain, root_call=True)
        return cells

    def export_resolved_meshes(self, directory: str | Path) -> dict[str, Optional[str]]:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = self._mesh_resolved_instances()
        outputs: dict[str, Optional[str]] = {}
        for label, mesh in resolved.items():
            safe_name = label.replace("/", "__") + ".stl"
            if mesh is not None and not mesh.is_empty:
                path = out_dir / safe_name
                mesh.export(path)
                outputs[label] = str(path)
            else:
                outputs[label] = None
        return outputs

    def export_combined_mesh(self, path: str | Path, *, resolved: bool = True) -> trimesh.Trimesh:
        trimesh = _require_trimesh()
        if resolved:
            parts = [m for m in self._mesh_resolved_instances().values()
                     if m is not None and not m.is_empty]
        else:
            parts = []
            for instance in self.iter_instances():
                mesh = instance.world_mesh()
                if mesh is not None:
                    parts.append(mesh)
        combined = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()
        combined.export(path)
        return combined

    @staticmethod
    def _normalize_instance_path(instance_path) -> str:
        if isinstance(instance_path, str):
            return instance_path.strip("/")
        return "/".join(str(part) for part in instance_path)

    def export_instance_stl(self, instance_path: str | tuple[str, ...],
                            filename: str | Path, *, resolved: bool = True) -> Path:
        """Export one placed instance in world coordinates.

        If ``resolved`` is true, sibling and parent/child priority rules are
        applied using the preview mesh boolean backend. This is approximate
        and may require ``manifold3d``.
        """
        _require_trimesh()
        label = self._normalize_instance_path(instance_path)
        instance = next((item for item in self.iter_instances() if item.label == label), None)
        if instance is None:
            raise KeyError(f"No scene instance has path '{label}'")
        if resolved:
            mesh = self._mesh_resolved_instances().get(label)
        else:
            mesh = instance.world_mesh()
        if mesh is None or mesh.is_empty:
            raise ValueError(f"Instance '{label}' has no exportable mesh")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        return path

    def export_node_stl(self, node_path: str | tuple[str, ...],
                        filename: str | Path, *, resolved: bool = True) -> Path:
        """Export a node and all descendants as one STL mesh in world coordinates."""
        _require_trimesh()
        prefix = self._normalize_instance_path(node_path)
        instances = list(self.iter_instances())
        selected = [item for item in instances
                    if item.label == prefix or item.label.startswith(prefix + "/")]
        if not selected:
            raise KeyError(f"No scene node matches instance path '{prefix}'")
        if resolved:
            mesh_map = self._mesh_resolved_instances()
            parts = [mesh_map[item.label] for item in selected
                     if mesh_map.get(item.label) is not None
                     and not mesh_map[item.label].is_empty]
        else:
            parts = []
            for item in selected:
                mesh = item.world_mesh()
                if mesh is not None and not mesh.is_empty:
                    parts.append(mesh)
        if not parts:
            raise ValueError(f"Node '{prefix}' has no exportable mesh")
        trimesh = _require_trimesh()
        combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(path)
        return path

    def export_volume_stl(self, volume: str | VolumeDef, filename: str | Path, *,
                          instance_path: str | tuple[str, ...] | None = None,
                          include_children: bool = False, resolved: bool = True) -> Path:
        """Export a definition occurrence, or its complete subtree, as STL.

        A reused VolumeDef may have several occurrences. In that case provide
        ``instance_path`` to select the desired occurrence.
        """
        definition = self.defs[volume] if isinstance(volume, str) else volume
        if instance_path is None:
            matches = [item for item in self.iter_instances()
                       if item.volume_def is definition]
            if len(matches) != 1:
                labels = [item.label for item in matches]
                raise ValueError(
                    f"Volume '{definition.name}' has {len(matches)} occurrences; "
                    f"select one with instance_path. Available paths: {labels}"
                )
            label = matches[0].label
        else:
            label = self._normalize_instance_path(instance_path)
            match = next((item for item in self.iter_instances()
                          if item.label == label and item.volume_def is definition), None)
            if match is None:
                raise KeyError(
                    f"Instance path '{label}' does not refer to VolumeDef '{definition.name}'"
                )
        if include_children:
            return self.export_node_stl(label, filename, resolved=resolved)
        return self.export_instance_stl(label, filename, resolved=resolved)

    def export_stl(self, filename: str | Path, *, resolved: bool = True) -> trimesh.Trimesh:
        """Export the complete scene as one STL mesh."""
        return self.export_combined_mesh(filename, resolved=resolved)

    def view(self, resolved: bool = True, outer_opacity: float = 0.25,
             inner_opacity: float = 0.90, interactive_clip: bool = False) -> None:
        _require_trimesh()
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "Interactive visualization requires PyVista. Install it with "
                "`pip install minigeom-analytic[visualization]`."
            ) from exc
        instances = list(self.iter_instances())
        resolved_meshes = self._mesh_resolved_instances() if resolved else {}
        rng = np.random.default_rng(42)
        colors: dict[str, str] = {}
        plotter = pv.Plotter(window_size=(1100, 800))
        plotter.set_background("white")
        added = False
        for index, inst in enumerate(instances):
            mesh = resolved_meshes.get(inst.label) if resolved else inst.world_mesh()
            if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
                continue
            pv_mesh = _tri_to_pv(mesh)
            if pv_mesh.n_points == 0:
                continue
            material_label = inst.material if inst.material is not None else "<unassigned>"
            if material_label not in colors:
                rgb = rng.integers(50, 220, size=3)
                colors[material_label] = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            opacity = outer_opacity if index == 0 else inner_opacity
            if interactive_clip:
                plotter.add_mesh_clip_plane(pv_mesh, color=colors[material_label],
                                            opacity=opacity, assign_to_axis="z")
            else:
                plotter.add_mesh(pv_mesh, color=colors[material_label], opacity=opacity,
                                 label=f"{inst.label} ({material_label})", show_edges=False)
            added = True
        if not added:
            print("No non-empty mesh is available to display.")
            return
        plotter.add_axes()
        if not interactive_clip:
            plotter.add_legend(bcolor="white")
        plotter.show()
