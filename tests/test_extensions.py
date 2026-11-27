import numpy as np
import pytest

from minigeom import (
    Box, Cone, Cylinder, Ellipsoid, Plane, Quadric, Scene, Sphere, Torus,
    CSGCompiler, OverlapChecker,
)


def material_at(model, point):
    hits = model.find_material(point)
    assert len(hits) == 1, (point, [(h.path, h.material) for h in hits])
    return hits[0].material


def make_world(scene_name="new_primitives"):
    scene = Scene(scene_name)
    world = scene.define("world", Box(30, 30, 30), material="air")
    return scene, world


def test_finite_frustum_cone_analytical_region():
    scene, world = make_world()
    cone = scene.define("cone", Cone(bottom_radius=3, top_radius=1, height=8), "steel")
    world.place(cone)
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0, 0, 0)) == "steel"
    assert material_at(model, (2.5, 0, -3)) == "steel"
    assert material_at(model, (2.5, 0, 3)) == "air"
    assert any(s.kind == "quadric" for s in model.surfaces)


def test_rotated_ellipsoid_uses_general_quadric():
    scene, world = make_world()
    ellipsoid = scene.define("ellipsoid", Ellipsoid((4, 1, 1)), "fuel")
    world.place(ellipsoid, rotate=(0, 90, 0))
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0, 0, 3.5)) == "fuel"
    assert material_at(model, (3.5, 0, 0)) == "air"
    assert any(s.kind == "quadric" for s in model.surfaces)


def test_plane_halfspace_can_be_clipped_by_world():
    scene, world = make_world()
    half = scene.define("half", Plane((0, 0, 1), offset=0, sense=-1), "water")
    world.place(half)
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0, 0, -2)) == "water"
    assert material_at(model, (0, 0, 2)) == "air"


def test_general_quadric_halfspace():
    scene, world = make_world()
    # Unit sphere x²+y²+z²-4 <= 0, represented as a general quadric.
    q = scene.define("quadric_sphere", Quadric((1,1,1,0,0,0,0,0,0,-4)), "ceramic")
    world.place(q)
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0, 0, 0)) == "ceramic"
    assert material_at(model, (3, 0, 0)) == "air"


def test_torus_analytical_region_and_rotated_axis():
    scene, world = make_world()
    ring = scene.define("ring", Torus(major_radius=4, minor_radius=1, axis="z"), "copper")
    world.place(ring, rotate=(90, 0, 0))  # local z-axis maps to world -y
    model = CSGCompiler().compile(scene)
    assert material_at(model, (4, 0, 0)) == "copper"
    assert material_at(model, (0, 0, 0)) == "air"
    torus = next(s for s in model.surfaces if s.kind == "torus")
    assert np.allclose(torus.coefficients[-3:], (0, -1, 0))


def test_unbounded_primitives_cannot_be_world_root():
    scene = Scene("bad_root")
    scene.define("root", Plane((0, 0, 1)), material="air")
    with pytest.raises(ValueError, match="bounded solid"):
        CSGCompiler().compile(scene)


def test_quadric_root_requires_explicit_aabb():
    coeff = (1,1,1,0,0,0,0,0,0,-4)
    with pytest.raises(ValueError, match="explicit AABB"):
        Quadric(coeff, bounded=True)
    scene = Scene("quadric_root")
    scene.define("root", Quadric(coeff, bounded=True,
                                  bounds=((-2,-2,-2),(2,2,2))), "air")
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0,0,0)) == "air"


def test_overlap_checker_reports_raw_and_resolved_sibling_overlap():
    scene, world = make_world("overlap_check")
    a = scene.define("a", Sphere(3), "a_mat")
    b = scene.define("b", Sphere(3), "b_mat")
    world.place(a, at=(-1, 0, 0), priority=1)
    world.place(b, at=(1, 0, 0), priority=5)
    report = OverlapChecker(samples=1200).check(scene)
    siblings = [p for p in report.detected_overlaps if p.relationship == "siblings"]
    assert len(siblings) == 1
    assert siblings[0].status == "resolved_by_priority"
    assert report.world_unassigned_points == 0
    assert report.world_multi_cell_points == 0
    assert not report.has_partition_issues


def test_overlap_checker_detects_partition_issue_when_model_is_manually_corrupted():
    scene, world = make_world("overlap_corrupt")
    ball = scene.define("ball", Sphere(2), "steel")
    world.place(ball)
    model = CSGCompiler().compile(scene)
    # Deliberately duplicate the resolved ball cell to emulate an exporter defect.
    from copy import copy
    duplicate = copy(model.cells[-1])
    duplicate.id = len(model.cells) + 1
    duplicate.name += "_duplicate"
    model.cells.append(duplicate)
    report = OverlapChecker(samples=512).check(model)
    assert report.world_multi_cell_points > 0


def test_explicit_lattice_expands_to_instances_for_csg_and_keeps_indices():
    scene, world = make_world("lattice")
    tile = scene.define("tile", Box(2, 2, 2), "moderator")
    fuel = scene.define("fuel", Sphere(0.5), "fuel")
    tile.place(fuel)
    world.place_lattice(tile, shape=(2, 3, 1), pitch=(2, 2, 2),
                        lower_left=(-2, -3, -1), name="pins")
    instances = list(scene.iter_instances())
    lattice_instances = [i for i in instances if i.lattice_name == "pins"]
    assert len(lattice_instances) == 6
    assert {i.lattice_index for i in lattice_instances} == {
        (i, j, 0) for i in range(2) for j in range(3)
    }
    model = CSGCompiler().compile(scene)
    assert material_at(model, (-1, -2, 0)) == "fuel"
    assert material_at(model, (1, 2, 0)) == "fuel"


def test_lattice_requires_box_tile_matching_pitch():
    scene, world = make_world("bad_lattice")
    pin = scene.define("pin", Sphere(1), "fuel")
    with pytest.raises(ValueError, match="Box"):
        world.place_lattice(pin, shape=(2,2,1), pitch=(2,2,2), lower_left=(-2,-2,-1))


def test_plane_normalization_preserves_original_equation():
    scene, world = make_world("plane_offset")
    # 2*z = 2 is the same plane as z = 1.
    half = scene.define("half", Plane((0, 0, 2), offset=2, sense=-1), "water")
    world.place(half)
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0, 0, 0.5)) == "water"
    assert material_at(model, (0, 0, 1.5)) == "air"


def test_bounded_quadric_aabb_is_part_of_region():
    scene = Scene("bounded_quadric")
    root = scene.define(
        "root",
        Quadric((1, 1, 1, 0, 0, 0, 0, 0, 0, -4), bounded=True,
                bounds=((-3, -3, -3), (3, 3, 3))),
        "air",
    )
    model = CSGCompiler().compile(scene)
    assert material_at(model, (0, 0, 0)) == "air"
    assert model.find_material((2.5, 0, 0)) == []
    assert len(model.surfaces) == 7  # one quadric + six explicit AABB planes


def test_volume_clone_preserves_reused_definition_identity():
    scene = Scene("clone_graph")
    root = scene.define("root", Box(20, 20, 20), "air")
    repeated = scene.define("pin", Box(2, 2, 2), "moderator")
    fuel = scene.define("fuel", Sphere(0.5), "fuel")
    repeated.place(fuel)
    root.place(repeated, at=(-3, 0, 0), name="left")
    root.place(repeated, at=(3, 0, 0), name="right")

    clone = root.clone(name="root_copy")
    assert clone.children[0].child is clone.children[1].child
    assert clone.children[0].child is not repeated


def test_lattice_replacement_api_and_instance_expansion():
    scene = Scene("substitution")
    world = scene.define("world", Box(12, 8, 2), material="water")
    tile = scene.define("default_tile", Box(2, 2, 2), material="water")
    pin = scene.define("pin", Cylinder(outer_radius=0.4, height=2), material="fuel")
    tile.place(pin)
    guide_tile = scene.define("guide_tile", Box(2, 2, 2), material="water")
    guide = scene.define("guide", Cylinder(outer_radius=0.6, height=2), material="steel")
    guide_tile.place(guide)
    lattice = world.place_lattice(tile, shape=(3, 2, 1), pitch=(2, 2, 2),
                                 lower_left=(-3, -2, -1), name="grid")
    assert lattice[0, 0, 0] is tile
    lattice[1, 0, 0] = guide_tile
    lattice.set_element(2, 1, 0, guide_tile)
    assert lattice.get_element(1, 0, 0) is guide_tile
    assert lattice[2, 1, 0] is guide_tile
    lattice.clear_element(2, 1, 0)
    assert lattice[2, 1, 0] is tile
    lattice[2, 1, 0] = guide_tile
    expanded = [inst for inst in scene.iter_instances() if inst.lattice_name == "grid"]
    by_index = {inst.lattice_index: inst for inst in expanded}
    assert by_index[(1, 0, 0)].volume_def is guide_tile
    assert by_index[(2, 1, 0)].volume_def is guide_tile
    assert by_index[(0, 0, 0)].volume_def is tile
    assert "guide_tile" in scene.defs


def test_lattice_replacement_index_validation():
    scene = Scene("invalid_index")
    world = scene.define("world", Box(6, 4, 2))
    tile = scene.define("tile", Box(2, 2, 2))
    replacement = scene.define("replacement", Box(2, 2, 2))
    lattice = world.place_lattice(tile, shape=(3, 2, 1), pitch=(2, 2, 2),
                                 lower_left=(-3, -2, -1))
    with pytest.raises(IndexError):
        lattice[3, 0, 0] = replacement
    with pytest.raises(IndexError):
        lattice[-1, 0, 0] = replacement
    with pytest.raises(TypeError):
        lattice[0.5, 0, 0] = replacement
    with pytest.raises(IndexError):
        lattice[0, 0] = replacement


def test_lattice_replacement_rejects_cycles():
    scene = Scene("replacement_cycle")
    parent = scene.define("parent", Box(4, 4, 2))
    tile = scene.define("tile", Box(2, 2, 2))
    lattice = parent.place_lattice(tile, shape=(2, 2, 1), pitch=(2, 2, 2),
                                  lower_left=(-2, -2, -1))
    child = scene.define("child", Box(2, 2, 2))
    child.place(parent)
    with pytest.raises(ValueError, match="cycle"):
        lattice[0, 0, 0] = child


def test_lattice_replacement_requires_matching_tile_box():
    scene = Scene("replacement_tile_validation")
    parent = scene.define("parent", Box(6, 4, 2))
    tile = scene.define("tile", Box(2, 2, 2))
    lattice = parent.place_lattice(tile, shape=(3, 2, 1), pitch=(2, 2, 2),
                                  lower_left=(-3, -2, -1))
    bad_size = scene.define("bad_size", Box(1, 2, 2))
    bad_shape = scene.define("bad_shape", Sphere(1))
    with pytest.raises(ValueError, match="match the lattice pitch"):
        lattice[0, 0, 0] = bad_size
    with pytest.raises(ValueError, match="must be a Box"):
        lattice[0, 0, 0] = bad_shape
