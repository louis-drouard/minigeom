"""TRIPOLI-5 exporter contract tests using a small AGORA API double."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import minigeom.tripoli5_exporter as tripoli_module
from minigeom import Box, Cylinder, Scene, Sphere, Tripoli5Exporter


class FakeRegion:
    def __init__(self, expression="region"):
        self.expression = expression

    def __and__(self, other):
        return FakeRegion(f"({self.expression}&{other.expression})")

    def __or__(self, other):
        return FakeRegion(f"({self.expression}|{other.expression})")

    def __neg__(self):
        return FakeRegion(f"-({self.expression})")


class FakeSurface:
    def __init__(self, *args, label=""):
        self.args = args
        self.label = label

    def __neg__(self):
        return FakeRegion(f"-{self.label}")

    def __pos__(self):
        return FakeRegion(f"+{self.label}")


class FakeShape:
    def __init__(self, region):
        self.region = region


class FakeVolume:
    def __init__(self, shape, material, label=""):
        self.shape, self.material, self.label = shape, material, label


class FakeCompositeVolume:
    def __init__(self, shape, filler, affinetransformation=None, label=""):
        self.shape = shape
        self.filler = filler
        self.affinetransformation = affinetransformation
        self.label = label


class FakeUniverse:
    def __init__(self, volumes, label=""):
        self.volumes = list(volumes)
        self.label = label


class FakeRectLattice:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeBoundaryConditions:
    def __init__(self):
        self.entries = []

    def add(self, surface, condition):
        self.entries.append((surface, condition))


class FakeBoundaryCondition:
    pass


class FakeGeometry:
    def __init__(self, root_universe, boundary_conditions=None, description=""):
        self.root_universe = root_universe
        self.boundary_conditions = boundary_conditions
        self.description = description

    def serialize(self, filename):
        Path(filename).write_bytes(b"fake-agora-geometry")


class FakePoint:
    def __init__(self, *values):
        self.values = tuple(values)


class FakeVector:
    def __init__(self, *values):
        self.values = tuple(values)


class FakeTranslation:
    def __init__(self, vector):
        self.vector = vector


class FakeRotation:
    def __init__(self, vector):
        self.vector = vector


class FakeAffineTransformation:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation


def make_fake_tripoli():
    agora = SimpleNamespace(
        Sphere=FakeSurface,
        Plane=FakeSurface,
        Quadric=FakeSurface,
        Shape=FakeShape,
        Volume=FakeVolume,
        CompositeVolume=FakeCompositeVolume,
        Universe=FakeUniverse,
        RectLattice=FakeRectLattice,
        BoundaryConditions=FakeBoundaryConditions,
        LeakageBoundaryCondition=FakeBoundaryCondition,
        ReflectionBoundaryCondition=FakeBoundaryCondition,
        Geometry=FakeGeometry,
    )
    return agora


def patch_tripoli(monkeypatch):
    monkeypatch.setattr(tripoli_module, "agora", make_fake_tripoli())
    monkeypatch.setattr(tripoli_module, "Point", FakePoint)
    monkeypatch.setattr(tripoli_module, "Vector", FakeVector)
    monkeypatch.setattr(tripoli_module, "Translation", FakeTranslation)
    monkeypatch.setattr(tripoli_module, "Rotation", FakeRotation)
    monkeypatch.setattr(tripoli_module, "AffineTransformation", FakeAffineTransformation)


def test_tripoli_hierarchy_preserves_material_names_and_exports_binary(monkeypatch, tmp_path):
    patch_tripoli(monkeypatch)
    scene = Scene("tripoli_simple")
    world = scene.define("world", Box(20, 20, 20), material="WATER")
    insert = scene.define("insert", Sphere(2), material="STEEL")
    world.place(insert, at=(1, 0, 0), rotate=(0, 0, 90))

    exporter = Tripoli5Exporter(scene, mode="hierarchy")
    output = exporter.export(tmp_path / "geometry.agora")

    assert output.exists()
    assert output.read_bytes() == b"fake-agora-geometry"
    composite = next(v for v in exporter.volumes if isinstance(v, FakeCompositeVolume))
    leaf_materials = [v.material for v in exporter.volumes if isinstance(v, FakeVolume)]
    assert "WATER" in leaf_materials
    assert composite.filler is not None
    # AGORA translates first and rotates second. The pre-rotation translation
    # is R.T @ (1, 0, 0) for MiniGeom's R @ p + t convention.
    assert np.allclose(composite.affinetransformation.translation.vector.values,
                       (0, -1, 0), atol=1e-10)
    assert composite.affinetransformation.rotation.vector.values == (0.0, 0.0, 90.0)


def test_tripoli_hierarchy_preserves_native_rectangular_lattice(monkeypatch, tmp_path):
    patch_tripoli(monkeypatch)
    scene = Scene("tripoli_lattice")
    world = scene.define("world", Box(20, 20, 10), material="WATER")
    tile = scene.define("tile", Box(2, 2, 2), material="MODERATOR")
    pin = scene.define("pin", Cylinder(0.5, 2), material="FUEL")
    tile.place(pin)
    world.place_lattice(tile, shape=(3, 2, 1), pitch=(2, 2, 2),
                        lower_left=(-3, -2, -1), name="pins")

    exporter = Tripoli5Exporter(scene, mode="hierarchy")
    exporter.build()

    assert len(exporter.lattices) == 1
    lattice = exporter.lattices["pins"]
    assert lattice.kwargs["nx"] == 3
    assert lattice.kwargs["ny"] == 2
    assert lattice.kwargs["nz"] == 1
    assert lattice.kwargs["outer_universe"] is None


def test_tripoli_requires_a_material_tag_for_every_leaf(monkeypatch):
    patch_tripoli(monkeypatch)
    scene = Scene("tripoli_missing_material")
    scene.define("world", Box(10, 10, 10))
    exporter = Tripoli5Exporter(scene, mode="hierarchy")
    with pytest.raises(ValueError, match="material-name string"):
        exporter.build()


def test_tripoli_default_material_fills_untagged_volumes(monkeypatch):
    patch_tripoli(monkeypatch)
    scene = Scene("tripoli_default_material")
    scene.define("world", Box(10, 10, 10))
    exporter = Tripoli5Exporter(scene, default_material="VOID_BG", mode="hierarchy")
    exporter.build()
    assert any(v.material == "VOID_BG" for v in exporter.volumes if isinstance(v, FakeVolume))
