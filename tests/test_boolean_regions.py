"""Tests for user-facing analytical Boolean expressions and explicit cells."""

import pytest

from minigeom import (
    Box,
    CSGCompiler,
    Cylinder,
    RegionExpression,
    Scene,
    Sphere,
    intersection_all,
    union_all,
)


def single_material(model, point):
    cells = model.find_cells(point)
    assert len(cells) == 1, (point, [(cell.path, cell.material) for cell in cells])
    return cells[0].material


def test_primitive_operators_create_lazy_boolean_expression():
    outer = Cylinder(outer_radius=5.0, height=10.0)
    inner = Cylinder(outer_radius=3.0, height=10.0)
    shell = outer - inner

    assert isinstance(shell, RegionExpression)
    assert shell.to_dict()["op"] == "difference"


def test_boolean_operators_compile_to_analytical_regions():
    scene = Scene("boolean_operators")
    world = scene.define("world", Box(20, 20, 20))

    world.add_cell("shell", Cylinder(outer_radius=5, height=10) - Cylinder(outer_radius=3, height=10), "steel")
    world.add_cell("core", Sphere(2), "fuel", priority=10)

    model = CSGCompiler().compile(scene)
    assert single_material(model, (4, 0, 0)) == "steel"
    assert single_material(model, (0, 0, 0)) == "fuel"
    assert model.find_cells((0, 0, 7)) == []


def test_union_and_intersection_helpers():
    scene = Scene("union_intersection")
    world = scene.define("world", Box(20, 20, 20))

    left = Sphere(2).translated((-2, 0, 0))
    right = Sphere(2).translated((2, 0, 0))
    union = union_all([left, right])
    clipping = Box(6, 10, 10)

    world.add_cell("clipped_union", union & clipping, "material")
    model = CSGCompiler().compile(scene)
    assert single_material(model, (-2, 0, 0)) == "material"
    assert single_material(model, (2, 0, 0)) == "material"
    # The owner has no material tag, so points outside explicit cells remain unassigned.
    assert model.find_cells((0, 1, 0)) == []

    # intersection_all is an alternative spelling for repeated '&' operators.
    expr = intersection_all([Sphere(4), Box(4, 4, 4)])
    assert isinstance(expr, RegionExpression)
    assert expr.to_dict()["op"] == "intersection"


def test_complement_is_clipped_to_the_owning_universe():
    scene = Scene("complement_clipping")
    world = scene.define("world", Box(10, 10, 10))
    # The complement of a sphere is mathematically unbounded. add_cell clips
    # the expression to the finite root Box, so the exported cell is bounded.
    world.add_cell("outside_sphere", ~Sphere(2), "outside")
    model = CSGCompiler().compile(scene)
    assert single_material(model, (4, 0, 0)) == "outside"
    assert model.find_cells((0, 0, 0)) == []
    assert model.find_cells((20, 0, 0)) == []


def test_region_expression_transform_is_composed_before_csg_compilation():
    scene = Scene("expression_transform")
    world = scene.define("world", Box(20, 20, 20))
    world.add_cell(
        "shifted_sphere",
        Sphere(1).rotated((0, 0, 45)).translated((3, 0, 0)),
        "steel",
    )
    model = CSGCompiler().compile(scene)
    assert single_material(model, (3, 0, 0)) == "steel"
    assert model.find_cells((0, 0, 0)) == []


def test_explicit_region_cell_priority_higher_wins_and_last_wins_ties():
    scene = Scene("explicit_priorities")
    world = scene.define("world", Box(20, 20, 20))
    world.add_cell("low", Sphere(3), "low", priority=1)
    world.add_cell("high", Sphere(2).translated((1, 0, 0)), "high", priority=10)

    model = CSGCompiler().compile(scene)
    assert single_material(model, (1, 0, 0)) == "high"
    assert single_material(model, (-2.5, 0, 0)) == "low"

    tie_scene = Scene("explicit_ties")
    tie_world = tie_scene.define("world", Box(10, 10, 10))
    tie_world.add_cell("first", Sphere(2), "first", priority=5)
    tie_world.add_cell("second", Sphere(2), "second", priority=5)
    tie_model = CSGCompiler().compile(tie_scene)
    assert single_material(tie_model, (0, 0, 0)) == "second"


def test_explicit_region_cells_can_be_attached_to_a_placed_volume():
    scene = Scene("placed_region_cells")
    world = scene.define("world", Box(30, 30, 30), "air")
    insert = scene.define("insert", Cylinder(outer_radius=5, height=10), material=None)
    insert.add_cell("core", Cylinder(outer_radius=2, height=10), "fuel")
    insert.add_cell("annulus", Cylinder(outer_radius=5, height=10) - Cylinder(outer_radius=2, height=10), "steel")
    world.place(insert, at=(5, 0, 0))

    model = CSGCompiler().compile(scene)
    assert single_material(model, (5, 0, 0)) == "fuel"
    assert single_material(model, (8, 0, 0)) == "steel"
    assert single_material(model, (0, 4, 0)) == "air"


def test_explicit_cells_cannot_be_mixed_with_implicit_placements():
    scene = Scene("mixed_partition")
    world = scene.define("world", Box(10, 10, 10))
    child = scene.define("child", Sphere(1), "steel")
    world.add_cell("explicit", Sphere(2), "water")
    with pytest.raises(ValueError, match="cannot be mixed"):
        world.place(child)

    other = scene.define("other", Box(10, 10, 10))
    other.place(child)
    with pytest.raises(ValueError, match="cannot be mixed"):
        other.add_cell("explicit", Sphere(2), "water")


def test_union_all_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        union_all([])


def test_overlap_checker_includes_raw_explicit_cell_footprints():
    scene = Scene("explicit_overlap_diagnostic")
    world = scene.define("world", Box(20, 20, 20), material="background")
    world.add_cell("first", Sphere(3), "first", priority=1)
    world.add_cell("second", Sphere(3).translated((1, 0, 0)), "second", priority=2)

    report = scene.check_overlaps(samples=256)
    explicit_pair = next(
        pair for pair in report.pair_results
        if {pair.first.rsplit("/", 1)[-1], pair.second.rsplit("/", 1)[-1]} == {"first", "second"}
    )
    assert explicit_pair.detected
    assert explicit_pair.resolved
    assert explicit_pair.status == "resolved_by_priority"
