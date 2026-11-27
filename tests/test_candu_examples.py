"""Both CANDU examples must describe the same representative material layout."""
import importlib.util
from pathlib import Path

from minigeom import CSGCompiler


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _material_at(model, point):
    cells = model.find_cells(point)
    assert len(cells) == 1, (point, [(cell.path, cell.material) for cell in cells])
    return cells[0].material


def test_hierarchical_and_boolean_candu_examples_have_same_material_probes():
    hierarchical = _load_example(
        "candu_hierarchical_example", EXAMPLES / "candu_hierarchical.py"
    )
    boolean = _load_example(
        "candu_boolean_example", EXAMPLES / "candu_boolean_regions.py"
    )

    hierarchy_scene = hierarchical.scene
    boolean_scene = boolean.scene
    hierarchy_model = CSGCompiler(boundary_type="reflective").compile(hierarchy_scene)
    boolean_model = CSGCompiler(boundary_type="reflective").compile(boolean_scene)

    assert sum(
        instance.volume_def.name == "fuel_pin"
        for instance in hierarchy_scene.iter_instances()
    ) == 37
    assert len(boolean_scene.top_volume.region_cells) == 10
    # The hierarchy retains one resolved cell per material-bearing instance;
    # the explicit CSG version groups the 37 fuel and cladding components into
    # two disconnected union cells, giving a much smaller flat cell list.
    assert len(hierarchy_model.cells) == 82
    assert len(boolean_model.cells) == 10

    # A deterministic Halton sample checks that the ten explicit regions tile
    # the assembly domain: no sampled point should be left unassigned or belong
    # to several resolved cells after the compiler's Boolean arbitration.
    report = boolean_scene.check_overlaps(samples=1024)
    assert not report.has_partition_issues, report.summary()

    probes = [
        (0.0, 0.0, 0.0),
        (0.63, 0.0, 0.0),
        (0.70, 0.0, 0.0),
        (1.4880, 0.80, 0.0),
        (5.40, 0.0, 0.0),
        (6.00, 0.0, 0.0),
        (6.50, 0.0, 0.0),
        (7.00, 0.0, 0.0),
    ]
    for point in probes:
        assert _material_at(hierarchy_model, point) == _material_at(boolean_model, point)

    assert len(hierarchy_model.surfaces) == len(boolean_model.surfaces) == 87
