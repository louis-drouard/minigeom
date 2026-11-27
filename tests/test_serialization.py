import zipfile

import numpy as np
import pytest

from minigeom import Box, Cylinder, Scene, CSGCompiler, save_volume, load_volume



def build_archiveable_graph():
    scene = Scene("archive-test")
    root = scene.define("assembly", Box(10, 10, 20), material="water")
    tile = scene.define("tile", Box(2, 2, 20), material="water")
    pin_tile = scene.define("pin_tile", Box(2, 2, 20), material="water")
    pin = scene.define("pin", Cylinder(outer_radius=0.6, height=20), material="steel")
    fuel = scene.define("fuel", Cylinder(outer_radius=0.4, height=20), material="fuel")
    pin.place(fuel, priority=10, name="fuel_region")
    pin_tile.place(pin, at=(0, 0, 0), priority=20, name="pin_center")
    root.place(pin_tile, at=(0, 0, 0), name="reused_pin_tile")
    lattice = root.place_lattice(tile, shape=(5, 5, 1), pitch=(2, 2, 20),
                                lower_left=(-5, -5, -10), name="grid")
    lattice[2, 2, 0] = pin_tile
    # Explicit Boolean regions can coexist in separately archived definitions.
    return scene, root



def test_archive_roundtrip_preserves_shared_refs_and_sparse_lattice(tmp_path):
    scene, root = build_archiveable_graph()
    path = scene.save_volume(root, tmp_path / "assembly")
    assert path.suffix == ".mgz"
    restored = load_volume(path)
    restored_scene = Scene("restored")
    restored_scene.set_top(restored)
    lattice = restored.lattices[0]
    assert lattice.shape == (5, 5, 1)
    assert lattice.pitch == (2.0, 2.0, 20.0)
    assert len(lattice.replacements) == 1
    assert lattice[2, 2, 0].name == "pin_tile"
    # Shared pin definition is retained rather than duplicated.
    assert lattice[2, 2, 0] is lattice.replacements[(2, 2, 0)]
    # The direct placement and sparse replacement point to the same definition.
    assert restored.children[0].child is lattice.replacements[(2, 2, 0)]
    compiled = CSGCompiler().compile(restored_scene)
    original_compiled = CSGCompiler().compile(scene)
    assert len(compiled.cells) == len(original_compiled.cells)
    assert len(compiled.surfaces) == len(original_compiled.surfaces)


def test_archive_roundtrip_explicit_boolean_regions(tmp_path):
    scene = Scene("regions")
    volume = scene.define("cell", Box(10, 10, 10), material="water")
    volume.add_cell("annulus", Cylinder(outer_radius=3, height=10) - Cylinder(outer_radius=1, height=10), material="steel", priority=3)
    path = save_volume(volume, tmp_path / "region.mgz")
    loaded = load_volume(path)
    assert loaded.region_cells[0].name == "annulus"
    assert loaded.region_cells[0].material == "steel"
    assert loaded.region_cells[0].priority == 3
    assert loaded.region_cells[0].region.to_dict() == volume.region_cells[0].region.to_dict()


def test_archive_rejects_bad_format_and_version(tmp_path):
    path = tmp_path / "bad.mgz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("geometry.json", '{"format":"wrong"}')
    with pytest.raises(ValueError, match="Not a MiniGeom archive"):
        load_volume(path)


def test_archive_root_can_be_renamed(tmp_path):
    _, root = build_archiveable_graph()
    path = save_volume(root, tmp_path / "original.mgz")
    loaded = load_volume(path, name="loaded_assembly")
    assert loaded.name == "loaded_assembly"


def test_archive_roundtrip_preserves_primitive_parameters(tmp_path):
    from minigeom import Cone, Ellipsoid, Plane, Quadric, Sphere, Torus
    shapes = [
        Sphere(2.5, subdivisions=4),
        Cone(1.0, 0.5, 3.0, sections=12),
        Ellipsoid((1.0, 2.0, 3.0), subdivisions=4),
        Torus(4.0, 0.8, axis="x", major_sections=20, minor_sections=12),
        Plane((1, 2, 3), offset=4.0, sense=1),
        Quadric((1, 1, 1, 0, 0, 0, 0, 0, 0, -1), sense=-1, bounded=True,
                bounds=((-1, -1, -1), (1, 1, 1))),
    ]
    scene = Scene("primitives")
    root = scene.define("root", Box(20, 20, 20), material="water")
    for i, shape in enumerate(shapes):
        root.place(scene.define(f"shape_{i}", shape, material="mat"), name=f"shape_{i}")
    archive_path = save_volume(root, tmp_path / "primitives.mgz")
    loaded = load_volume(archive_path)
    for original, restored_placement in zip(shapes, loaded.children):
        restored_shape = restored_placement.child.shape
        assert type(restored_shape) is type(original)
        assert vars(restored_shape) == vars(original)


def test_scene_stl_export_remains_available_for_full_scene(tmp_path):
    pytest.importorskip("trimesh")
    scene = Scene("scene_stl")
    root = scene.define("root", Box(5, 5, 5), material="water")
    scene.define("sphere", __import__("minigeom").Sphere(1), material="fuel")
    root.place(scene.defs["sphere"], at=(1, 0, 0))
    scene.export_stl(tmp_path / "scene.stl", resolved=False)
    target = tmp_path / "scene.stl"
    assert target.exists()
    assert target.stat().st_size > 0


def test_scene_load_volume_namespaces_internal_definitions_to_allow_modular_imports(tmp_path):
    first_scene = Scene("component_a")
    first_root = first_scene.define("component_a", Box(10, 10, 10), material="water")
    first_child = first_scene.define("fuel", Cylinder(outer_radius=1, height=10), material="fuel")
    first_root.place(first_child, name="fuel_placement")
    path_a = first_scene.save_volume(first_root, tmp_path / "a.mgz")

    second_scene = Scene("component_b")
    second_root = second_scene.define("component_b", Box(10, 10, 10), material="water")
    second_child = second_scene.define("fuel", Cylinder(outer_radius=1, height=10), material="fuel")
    second_root.place(second_child, name="fuel_placement")
    path_b = second_scene.save_volume(second_root, tmp_path / "b.mgz")

    final_scene = Scene("final")
    loaded_a = final_scene.load_volume(path_a)
    loaded_b = final_scene.load_volume(path_b)
    assert loaded_a.name == "component_a"
    assert loaded_b.name == "component_b"
    assert "component_a__fuel" in final_scene.defs
    assert "component_b__fuel" in final_scene.defs
    assert loaded_a.children[0].child.name == "component_a__fuel"
    assert loaded_b.children[0].child.name == "component_b__fuel"


def test_annular_cylinder_archive_roundtrip(tmp_path):
    scene = Scene("annular_archive")
    root = scene.define("world", Box(10, 10, 20), material="water")
    annulus = scene.define("tube", Cylinder(outer_radius=2.0, inner_radius=1.25, height=10.0), material="steel")
    root.place(annulus)
    path = save_volume(root, tmp_path / "tube.mgz")
    loaded = load_volume(path)
    assert loaded.children[0].child.shape.outer_radius == 2.0
    assert loaded.children[0].child.shape.inner_radius == 1.25


def test_legacy_cylinder_archive_radius_field_loads_as_outer_radius():
    from minigeom.serialization import _primitive_from_data
    shape = _primitive_from_data({"type": "Cylinder", "radius": 2.0, "height": 5.0, "sections": 12})
    assert shape.outer_radius == 2.0
    assert shape.inner_radius == 0.0
