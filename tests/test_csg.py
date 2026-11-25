import json

import pytest

from minigeom import Box, Cylinder, Scene, Sphere, CSGCompiler


def locate(model, point):
    hits = model.find_material(point)
    assert len(hits) == 1, (point, [(h.path, h.material) for h in hits])
    return hits[0].material


def test_nested_hierarchy_and_material_partition():
    scene = Scene("nested")
    world = scene.define("world", Box(20, 20, 20), material="air")
    shield = scene.define("shield", Sphere(4), material="lead")
    target = scene.define("target", Sphere(1), material="tungsten")
    world.place(shield, priority=10)
    shield.place(target, priority=0)

    model = CSGCompiler().compile(scene)
    assert locate(model, (0, 0, 0)) == "tungsten"
    assert locate(model, (2, 0, 0)) == "lead"
    assert locate(model, (7, 0, 0)) == "air"
    assert len(model.surfaces) == 8  # six world planes plus two sphere surfaces
    assert len(model.cells) == 3


def test_sibling_priority_higher_wins_and_tie_is_later_inserted():
    scene = Scene("siblings")
    world = scene.define("world", Box(20, 20, 20), material="air")
    low = scene.define("low", Sphere(2), material="low_mat")
    high = scene.define("high", Sphere(2), material="high_mat")
    world.place(low, at=(-1, 0, 0), priority=1)
    world.place(high, at=(1, 0, 0), priority=10)
    model = CSGCompiler().compile(scene)
    assert locate(model, (0, 0, 0)) == "high_mat"

    scene2 = Scene("ties")
    root = scene2.define("root", Box(20, 20, 20), material="air")
    first = scene2.define("first", Sphere(2), material="first_mat")
    second = scene2.define("second", Sphere(2), material="second_mat")
    root.place(first, priority=5)
    root.place(second, priority=5)
    tied = CSGCompiler().compile(scene2)
    assert locate(tied, (0, 0, 0)) == "second_mat"


def test_protrusion_escapes_parent_but_not_world():
    scene = Scene("protrusion")
    world = scene.define("world", Box(20, 20, 20), material="air")
    small_container = scene.define("container", Sphere(1), material="container_mat")
    protruder = scene.define("insert", Sphere(2), material="insert_mat")
    world.place(small_container)
    small_container.place_protruding(protruder, at=(3, 0, 0), overlap_priority=50)
    model = CSGCompiler().compile(scene)
    assert locate(model, (3, 0, 0)) == "insert_mat"
    assert locate(model, (0, 0, 0)) == "container_mat"
    assert locate(model, (8, 0, 0)) == "air"
    # All cells are clipped to the root world; there should be no cell at x=11.
    assert model.find_material((11, 0, 0)) == []


def test_multiple_protrusions_priority():
    scene = Scene("two_protrusions")
    world = scene.define("world", Box(30, 30, 30), material="air")
    logical_a = scene.define("logical_a", None, material="vacuum")
    logical_b = scene.define("logical_b", None, material="vacuum")
    a = scene.define("a", Sphere(3), material="a_mat")
    b = scene.define("b", Sphere(3), material="b_mat")
    world.place(logical_a)
    world.place(logical_b)
    logical_a.place_protruding(a, at=(-1, 0, 0), overlap_priority=1)
    logical_b.place_protruding(b, at=(1, 0, 0), overlap_priority=10)
    model = CSGCompiler().compile(scene)
    assert locate(model, (0, 0, 0)) == "b_mat"


def test_shape_less_group_does_not_create_infinite_cell():
    scene = Scene("group")
    world = scene.define("world", Box(10, 10, 10), material="air")
    group = scene.define("assembly", None, material="vacuum")
    ball = scene.define("ball", Sphere(1), material="steel")
    world.place(group)
    group.place(ball, at=(2, 0, 0))
    model = CSGCompiler().compile(scene)
    assert locate(model, (2, 0, 0)) == "steel"
    assert locate(model, (0, 0, 0)) == "air"
    assert all(cell.volume_name != "assembly" for cell in model.cells)


def test_rotated_finite_cylinder_is_analytic_quadric():
    scene = Scene("rotated_cylinder")
    world = scene.define("world", Box(20, 20, 20), material="air")
    cyl = scene.define("cyl", Cylinder(1, 8), material="steel")
    world.place(cyl, rotate=(0, 90, 0))
    model = CSGCompiler().compile(scene)
    assert locate(model, (3, 0, 0)) == "steel"
    assert locate(model, (0, 0, 3)) == "air"
    assert any(surface.kind == "quadric" for surface in model.surfaces)


def test_json_schema_is_full_analytic_description(tmp_path):
    scene = Scene("json")
    world = scene.define("world", Box(10, 10, 10), material="air")
    ball = scene.define("ball", Sphere(2), material="steel")
    world.place(ball, at=(1, 0, 0))
    model = CSGCompiler().compile(scene)
    path = model.export_json(tmp_path / "geometry.json")
    payload = json.loads(path.read_text())
    assert payload["schema"] == "minigeom-csg-1.2"
    assert payload["surfaces"]
    assert payload["cells"]
    assert payload["cells"][0]["region"]["op"] == "intersection"


def test_shape_less_material_is_rejected():
    scene = Scene("invalid_group")
    world = scene.define("world", Box(10, 10, 10), material="air")
    group = scene.define("group", None, material="steel")
    world.place(group)
    with pytest.raises(ValueError, match="has no primitive"):
        CSGCompiler().compile(scene)


def test_root_requires_bounded_shape():
    scene = Scene("no_world")
    root = scene.define("root", None, material="vacuum")
    with pytest.raises(ValueError, match="root volume must be a bounded solid"):
        CSGCompiler().compile(scene)


def test_root_boundary_is_marked_and_internal_boundaries_transmit():
    scene = Scene("boundaries")
    world = scene.define("world", Box(10, 10, 10), material="air")
    ball = scene.define("ball", Sphere(1), material="steel")
    world.place(ball)
    model = CSGCompiler(boundary_type="reflective").compile(scene)
    boundary_surfaces = [s for s in model.surfaces if s.boundary_type == "reflective"]
    transmission_surfaces = [s for s in model.surfaces if s.boundary_type == "transmission"]
    assert len(boundary_surfaces) == 6
    assert len(transmission_surfaces) == 1


def test_reused_volume_definition_creates_separate_instances():
    scene = Scene("repeated")
    world = scene.define("world", Box(20, 20, 20), material="air")
    pin = scene.define("pin", Cylinder(1, 5), material="steel")
    world.place(pin, at=(-3, 0, 0), name="pin_left")
    world.place(pin, at=(3, 0, 0), name="pin_right")
    model = CSGCompiler().compile(scene)
    assert locate(model, (-3, 0, 0)) == "steel"
    assert locate(model, (3, 0, 0)) == "steel"
    assert sum(cell.volume_name == "pin" for cell in model.cells) == 2


def test_material_tag_is_optional_and_preserved_as_none():
    scene = Scene("geometry_only")
    world = scene.define("world", Box(10, 10, 10))
    ball = scene.define("ball", Sphere(2))
    world.place(ball)

    assert world.material is None
    assert ball.material is None
    model = CSGCompiler().compile(scene)
    assert all(cell.material is None for cell in model.cells)
    assert len(model.find_cells((0, 0, 0))) == 1
    assert model.find_material((0, 0, 0)) == model.find_cells((0, 0, 0))


def test_symbolic_material_tags_are_not_erased_by_csg_compiler():
    scene = Scene("symbolic_tags")
    world = scene.define("world", Box(10, 10, 10), material="air")
    ball = scene.define("ball", Sphere(2), material="steel")
    world.place(ball)
    model = CSGCompiler().compile(scene)
    assert {cell.material for cell in model.cells} == {"air", "steel"}


def test_clone_can_explicitly_clear_material_tag():
    scene = Scene("clone_clear_material")
    ball = scene.define("ball", Sphere(1), material="steel")
    clone = ball.clone(material=None)
    assert clone.material is None
    assert ball.material == "steel"
