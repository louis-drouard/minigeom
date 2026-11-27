"""The analytical package must import and compile CSG without Trimesh."""
import os
from pathlib import Path
import subprocess
import sys


def test_core_import_and_csg_compile_when_trimesh_is_unavailable():
    project_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
original_import = builtins.__import__
def block_trimesh(name, *args, **kwargs):
    if name == "trimesh" or name.startswith("trimesh."):
        raise ModuleNotFoundError("Trimesh intentionally unavailable in this test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_trimesh

from minigeom import Box, CSGCompiler, Scene, Sphere
scene = Scene("no_mesh_dependency")
world = scene.define("world", Box(10, 10, 10))
world.add_cell("ball", Sphere(2), material="fuel")
model = CSGCompiler().compile(scene)
assert len(model.find_cells((0, 0, 0))) == 1
assert "trimesh" not in __import__("sys").modules
try:
    Sphere(1).mesh()
except ImportError as error:
    assert "minigeom-analytic[mesh]" in str(error)
else:
    raise AssertionError("A preview mesh should require the optional mesh extra")
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
