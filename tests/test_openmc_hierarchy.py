"""Exporter structural tests using a small OpenMC API double.

These tests validate our calls and object graph. They do not replace a run with
an installed OpenMC release and XML round-trip.
"""
from types import SimpleNamespace
import pytest
import numpy as np
import json
import xml.etree.ElementTree as ET

from minigeom import Box, Cylinder, OpenMCExporter, Scene, Sphere
import minigeom.openmc_exporter as exporter_module


class FakeRegion:
    def __init__(self, label="r"):
        self.label = label
    def __and__(self, other): return FakeRegion(f"({self.label}&{other.label})")
    def __or__(self, other): return FakeRegion(f"({self.label}|{other.label})")
    def __invert__(self): return FakeRegion(f"~{self.label}")


class FakeSurface:
    _next_id = 1
    def __init__(self, *args, name="", boundary_type="transmission", **kwargs):
        self.args, self.kwargs = args, kwargs
        self.name, self.boundary_type = name, boundary_type
        self.id = FakeSurface._next_id
        FakeSurface._next_id += 1
    def __neg__(self): return FakeRegion("-" + self.name)
    def __pos__(self): return FakeRegion("+" + self.name)


class FakeMaterial:
    _next_id = 1
    def __init__(self, name="material"):
        self.name = name
        self.id = FakeMaterial._next_id
        FakeMaterial._next_id += 1


class FakeMaterials(list):
    def export_to_xml(self, path):
        root = ET.Element("materials")
        for material in self:
            ET.SubElement(root, "material", id=str(material.id), name=material.name)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class FakeCell:
    _next_id = 1
    def __init__(self, name="", fill=None, region=None):
        self.id = FakeCell._next_id
        FakeCell._next_id += 1
        self.name, self.fill, self.region = name, fill, region
        self.translation = None
        self.rotation = None


class FakeUniverse:
    _next_id = 1
    def __init__(self, name=""):
        self.id = FakeUniverse._next_id
        FakeUniverse._next_id += 1
        self.name = name
        self.cells = []
    def add_cells(self, cells): self.cells.extend(cells)
    def add_cell(self, cell): self.cells.append(cell)


class FakeRectLattice:
    def __init__(self, name=""):
        self.name = name
        self.lower_left = None
        self.pitch = None
        self.universes = None
        self.outer = None


class FakeGeometry:
    def __init__(self, root, **kwargs):
        self.root_universe = root
        self.kwargs = kwargs

    def export_to_xml(self, path):
        root = ET.Element("geometry")
        visited_universes = set()
        visited_cells = set()

        def add_universe(universe):
            if id(universe) in visited_universes:
                return
            visited_universes.add(id(universe))
            for cell in universe.cells:
                if id(cell) in visited_cells:
                    continue
                visited_cells.add(id(cell))
                attrs = {"id": str(cell.id), "name": cell.name}
                if isinstance(cell.fill, FakeMaterial):
                    attrs["material"] = str(cell.fill.id)
                elif cell.fill is None:
                    attrs["material"] = "void"
                else:
                    attrs["fill"] = getattr(cell.fill, "name", "universe")
                ET.SubElement(root, "cell", attrs)
                if isinstance(cell.fill, FakeUniverse):
                    add_universe(cell.fill)
                elif isinstance(cell.fill, FakeRectLattice):
                    for child_universe in cell.fill.universes.flat:
                        add_universe(child_universe)
                    if cell.fill.outer is not None:
                        add_universe(cell.fill.outer)

        add_universe(self.root_universe)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def make_fake_openmc():
    return SimpleNamespace(
        Material=FakeMaterial, Materials=FakeMaterials,
        Sphere=FakeSurface, Plane=FakeSurface, Quadric=FakeSurface,
        XTorus=FakeSurface, YTorus=FakeSurface, ZTorus=FakeSurface,
        Cell=FakeCell, Universe=FakeUniverse, RectLattice=FakeRectLattice,
        Geometry=FakeGeometry,
    )


def reset_fake_ids():
    FakeSurface._next_id = FakeCell._next_id = FakeUniverse._next_id = FakeMaterial._next_id = 1


def test_repeated_volume_def_reuses_one_openmc_universe(monkeypatch):
    reset_fake_ids()
    fake = make_fake_openmc()
    monkeypatch.setattr(exporter_module, "openmc", fake)
    scene = Scene("repeated_pins")
    world = scene.define("world", Box(20,20,20), "air")
    pin = scene.define("pin", Box(2,2,2), "moderator")
    fuel = scene.define("fuel", Cylinder(0.5,2), "fuel")
    pin.place(fuel)
    world.place(pin, at=(-3,0,0), name="left")
    world.place(pin, at=(3,0,0), name="right")
    mats = {name: FakeMaterial(name) for name in ("air", "moderator", "fuel")}
    exporter = OpenMCExporter(scene, mats, mode="hierarchy")
    exporter.build()
    root_cells = exporter.geometry.root_universe.cells
    fills = [cell.fill for cell in root_cells if getattr(cell.fill, "name", "") == "u_pin"]
    assert len(fills) == 2
    assert fills[0] is fills[1]
    assert sum(u.name == "u_pin" for u in exporter.universes.values()) == 1


def test_rectangular_lattice_exports_one_lattice_and_reuses_tile_universe(monkeypatch):
    reset_fake_ids()
    fake = make_fake_openmc()
    monkeypatch.setattr(exporter_module, "openmc", fake)
    scene = Scene("pin_lattice")
    world = scene.define("world", Box(20,20,10), "air")
    tile = scene.define("tile", Box(2,2,2), "moderator")
    fuel = scene.define("fuel", Cylinder(0.5,2), "fuel")
    tile.place(fuel)
    world.place_lattice(tile, shape=(3,2,1), pitch=(2,2,2),
                        lower_left=(-3,-2,-1), name="assembly")
    mats = {name: FakeMaterial(name) for name in ("air", "moderator", "fuel")}
    exporter = OpenMCExporter(scene, mats, mode="hierarchy")
    exporter.build()
    assert len(exporter.lattices) == 1
    lattice = next(iter(exporter.lattices.values()))
    assert lattice.universes.shape == (1, 2, 3)
    assert all(lattice.universes.flat[0] is u for u in lattice.universes.flat)
    assert lattice.lower_left == (-3., -2., -1.)
    assert lattice.pitch == (2., 2., 2.)
    assert lattice.outer is not None


def test_auto_mode_falls_back_for_protruding_placements(monkeypatch):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("fallback")
    world = scene.define("world", Box(20,20,20), "air")
    holder = scene.define("holder", Sphere(1), "steel")
    insert = scene.define("insert", Sphere(2), "lead")
    world.place(holder)
    holder.place_protruding(insert, at=(2,0,0))
    mats = {name: FakeMaterial(name) for name in ("air", "steel", "lead")}
    exporter = OpenMCExporter(scene, mats, mode="auto")
    exporter.build()
    assert exporter._selected_mode == "flat"
    assert "protruding" in exporter._fallback_reason


def test_flat_export_maps_analytic_quadric_and_torus_surfaces(monkeypatch):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    from minigeom import Cone, Ellipsoid, Plane, Quadric, Torus

    scene = Scene("analytic_export")
    world = scene.define("world", Box(30, 30, 30), "air")
    cone = scene.define("cone", Cone(2.0, 1.0, 4.0), "lead")
    ellipsoid = scene.define("ellipsoid", Ellipsoid((3.0, 2.0, 1.0)), "steel")
    plane = scene.define("plane", Plane((0, 0, 1), offset=0), "water")
    quadric = scene.define("quadric", Quadric((1, 1, 1, 0, 0, 0, 0, 0, 0, -1)), "ceramic")
    torus = scene.define("torus", Torus(4.0, 1.0, axis="z"), "copper")
    world.place(cone, at=(-8, 0, 0))
    world.place(ellipsoid, at=(-2, 0, 0))
    world.place(plane, at=(4, 0, 0))
    world.place(quadric, at=(8, 0, 0))
    world.place(torus, at=(0, 8, 0))
    mats = {name: FakeMaterial(name) for name in
            ("air", "lead", "steel", "water", "ceramic", "copper")}

    exporter = OpenMCExporter(scene, mats, mode="flat")
    geometry = exporter.build()
    assert geometry.root_universe is not None
    assert any(spec.kind == "torus" for spec in exporter.model.surfaces)
    assert any(spec.kind == "plane" for spec in exporter.model.surfaces)
    assert any(spec.kind == "quadric" for spec in exporter.model.surfaces)


def test_flat_export_rejects_arbitrarily_rotated_torus_with_actionable_message(monkeypatch):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    from minigeom import Torus
    scene = Scene("tilted_torus")
    world = scene.define("world", Box(20, 20, 20), "air")
    ring = scene.define("ring", Torus(4.0, 1.0), "copper")
    world.place(ring, rotate=(23, 17, 11))
    mats = {"air": FakeMaterial("air"), "copper": FakeMaterial("copper")}
    exporter = OpenMCExporter(scene, mats, mode="flat")
    with pytest.raises(ValueError, match="axis that is not parallel"):
        exporter.build()


def test_geometry_only_export_writes_numeric_material_ids_without_fake_materials(monkeypatch, tmp_path):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("geometry_only_export")
    world = scene.define("world", Box(10, 10, 10))
    sphere = scene.define("sphere", Sphere(2), material="steel")
    world.place(sphere)

    exporter = OpenMCExporter(scene, mode="flat")
    exporter.export(tmp_path)

    assert exporter.material_ids == {"steel": 1}
    assert exporter.materials is None
    assert exporter._unbound_material_names == {"steel"}
    # No placeholder Material objects are created, so OpenMC's ID allocator is untouched.
    assert FakeMaterial._next_id == 1
    assert not (tmp_path / "materials.xml").exists()

    ids = json.loads((tmp_path / "materials_ids.json").read_text())
    assert ids["materials"]["steel"] == {"id": 1, "source": "external"}
    tree = ET.parse(tmp_path / "geometry.xml")
    cells = tree.getroot().findall("cell")
    assert any(cell.get("material") == "1" for cell in cells)
    assert all(cell.fill is None for cell in exporter.cells)
    assert {cell.material for cell in exporter.model.cells} == {None, "steel"}


def test_partial_material_map_writes_only_real_materials_xml(monkeypatch, tmp_path):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("partial_binding")
    world = scene.define("world", Box(10, 10, 10), material="air")
    sphere = scene.define("sphere", Sphere(2), material="steel")
    world.place(sphere)
    air = FakeMaterial("air")
    exporter = OpenMCExporter(scene, materials_map={"air": air}, mode="flat")
    exporter.export(tmp_path)

    assert exporter.material_ids == {"air": air.id, "steel": 1 if air.id != 1 else 2}
    assert exporter.materials is not None
    assert [mat.name for mat in exporter.materials] == ["air"]
    assert (tmp_path / "materials.xml").exists()
    ids = json.loads((tmp_path / "materials_ids.json").read_text())["materials"]
    assert ids["air"]["source"] == "openmc.Material"
    assert ids["steel"]["source"] == "external"
    xml = ET.parse(tmp_path / "geometry.xml").getroot()
    cell_materials = {cell.get("material") for cell in xml.findall("cell")}
    assert str(air.id) in cell_materials
    assert str(exporter.material_ids["steel"]) in cell_materials


def test_explicit_material_ids_are_respected(monkeypatch, tmp_path):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("explicit_ids")
    root = scene.define("world", Box(10, 10, 10), material="air")
    exporter = OpenMCExporter(scene, material_ids={"air": 42}, mode="flat")
    exporter.export(tmp_path)
    assert exporter.material_ids == {"air": 42}
    xml = ET.parse(tmp_path / "geometry.xml").getroot()
    assert xml.find("cell").get("material") == "42"


def test_partial_material_map_is_not_an_error(monkeypatch):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("partial_map")
    world = scene.define("world", Box(10, 10, 10))
    steel = scene.define("steel_part", Sphere(2), material="steel")
    world.place(steel)
    exporter = OpenMCExporter(scene, materials_map={}, mode="flat")
    exporter.build()
    assert exporter.material_ids == {"steel": 1}
    assert exporter._unbound_material_names == {"steel"}


def test_explicit_void_tag_maps_to_unfilled_openmc_cell(monkeypatch):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("explicit_void")
    world = scene.define("world", Box(10, 10, 10), material="void")
    exporter = OpenMCExporter(scene, mode="flat")
    exporter.build()
    assert len(exporter.cells) == 1
    assert exporter.cells[0].fill is None
    assert exporter._unbound_material_names == set()


def test_hierarchical_export_patches_symbolic_material_ids_in_nested_cells(monkeypatch, tmp_path):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("nested_symbolic_ids")
    world = scene.define("world", Box(20, 20, 20), material="water")
    pin = scene.define("pin", Box(4, 4, 4), material="moderator")
    fuel = scene.define("fuel", Cylinder(0.8, 4), material="uo2")
    pin.place(fuel)
    world.place(pin)

    exporter = OpenMCExporter(scene, mode="hierarchy")
    exporter.export(tmp_path)

    id_map = json.loads((tmp_path / "materials_ids.json").read_text())["materials"]
    assert set(id_map) == {"water", "moderator", "uo2"}
    xml_cells = ET.parse(tmp_path / "geometry.xml").getroot().findall("cell")
    material_refs = {cell.get("material") for cell in xml_cells}
    assert {str(item["id"]) for item in id_map.values()} <= material_refs
    assert len(xml_cells) >= 3
    assert not (tmp_path / "materials.xml").exists()


def test_hierarchical_lattice_export_patches_symbolic_material_ids(monkeypatch, tmp_path):
    reset_fake_ids()
    monkeypatch.setattr(exporter_module, "openmc", make_fake_openmc())
    scene = Scene("nested_lattice_symbolic_ids")
    world = scene.define("world", Box(12, 12, 6), material="water")
    tile = scene.define("tile", Box(2, 2, 2), material="moderator")
    fuel = scene.define("fuel", Cylinder(0.5, 2), material="uo2")
    tile.place(fuel)
    world.place_lattice(tile, shape=(3, 3, 1), pitch=(2, 2, 2),
                        lower_left=(-3, -3, -1), name="pins")

    exporter = OpenMCExporter(scene, mode="hierarchy")
    exporter.export(tmp_path)

    id_map = json.loads((tmp_path / "materials_ids.json").read_text())["materials"]
    xml_cells = ET.parse(tmp_path / "geometry.xml").getroot().findall("cell")
    material_refs = {cell.get("material") for cell in xml_cells}
    assert {str(item["id"]) for item in id_map.values()} <= material_refs
    assert len(exporter.lattices) == 1
    assert not (tmp_path / "materials.xml").exists()
