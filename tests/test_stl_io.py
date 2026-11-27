"""STL import/export tests for mesh-backed volumes and scene nodes."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from minigeom import Box, CSGCompiler, MeshPrimitive, Scene, Sphere


def make_stl(path: Path):
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    mesh.export(path)
    return mesh


def test_import_stl_creates_mesh_primitive_and_applies_scale(tmp_path):
    source = tmp_path / "part.stl"
    make_stl(source)

    scene = Scene("stl_import")
    part = scene.import_stl("part", source, scale=0.1)

    assert isinstance(part.shape, MeshPrimitive)
    assert part.shape.is_bounded
    assert np.allclose(part.shape.mesh().extents, (0.2, 0.4, 0.6), atol=1e-6)
    assert part.shape.source == str(source.resolve())


def test_import_stl_can_allow_an_open_surface_when_requested(tmp_path):
    source = tmp_path / "open_surface.stl"
    # Build an open surface by removing the two top triangles from a box.
    # Trimesh's slice_plane() currently imports scipy.spatial.cKDTree, while
    # SciPy is deliberately not a core or test dependency of MiniGeom.
    open_mesh = trimesh.creation.box()
    top_faces = open_mesh.face_normals[:, 2] > 0.9
    open_mesh.update_faces(~top_faces)
    open_mesh.remove_unreferenced_vertices()
    assert not open_mesh.is_watertight
    open_mesh.export(source)

    scene = Scene("open_mesh")
    part = scene.import_stl("open_part", source, require_watertight=False)
    assert isinstance(part.shape, MeshPrimitive)


def test_volume_and_scene_instance_stl_exports(tmp_path):
    source = tmp_path / "imported.stl"
    make_stl(source)

    scene = Scene("stl_scene")
    world = scene.define("world", Box(20, 20, 20), material="air")
    imported = scene.import_stl("imported", source, scale=0.5, material="steel")
    world.place(imported, at=(3, 0, 0), name="imported_copy")

    # Definition export remains in the primitive's local coordinates.
    definition_path = imported.export_stl(tmp_path / "definition.stl")
    definition_mesh = trimesh.load(definition_path, force="mesh")
    assert np.allclose(definition_mesh.bounds.mean(axis=0), (0, 0, 0), atol=1e-6)

    # Instance export applies its world transform.
    instance_path = scene.export_instance_stl(
        "world/imported_copy", tmp_path / "instance.stl", resolved=False
    )
    instance_mesh = trimesh.load(instance_path, force="mesh")
    assert np.allclose(instance_mesh.bounds.mean(axis=0), (3, 0, 0), atol=1e-6)

    # Node export combines the node and all descendants.
    node_path = scene.export_node_stl("world", tmp_path / "node.stl", resolved=False)
    node_mesh = trimesh.load(node_path, force="mesh")
    assert len(node_mesh.faces) > len(instance_mesh.faces)

    # A definition with several occurrences must be selected by instance path.
    world.place(imported, at=(-3, 0, 0), name="second_copy")
    with pytest.raises(ValueError, match="has 2 occurrences"):
        scene.export_volume_stl(imported, tmp_path / "ambiguous.stl", resolved=False)
    scene.export_volume_stl(
        imported, tmp_path / "selected.stl",
        instance_path="world/second_copy", resolved=False,
    )


def test_mesh_primitive_is_rejected_by_analytical_csg_export(tmp_path):
    source = tmp_path / "mesh.stl"
    make_stl(source)
    scene = Scene("mesh_csg_rejection")
    world = scene.define("world", Box(20, 20, 20), material="air")
    imported = scene.import_stl("imported", source)
    world.place(imported)

    with pytest.raises(NotImplementedError, match="cannot be compiled to analytical CSG"):
        CSGCompiler().compile(scene)


def test_missing_stl_file_has_an_english_exception(tmp_path):
    scene = Scene("missing_mesh")
    with pytest.raises(FileNotFoundError, match="Mesh file not found"):
        scene.import_stl("missing", tmp_path / "missing.stl")
