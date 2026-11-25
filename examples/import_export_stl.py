"""Import a CAD mesh as a scene volume and export its placed instance as STL."""
from minigeom import Box, Scene

scene = Scene("cad_scene")
world = scene.define("world", Box(100, 100, 100), material="air")
part = scene.import_stl(
    "bracket",
    "input/bracket.stl",
    material="steel",
    scale=0.1,  # Convert millimetres to centimetres, if required.
)
world.place(part, at=(10, 0, 0), name="bracket_instance")

# Export the mesh without applying boolean priority clipping.
scene.export_instance_stl(
    "world/bracket_instance",
    "bracket_in_world.stl",
    resolved=False,
)

# Export the complete world subtree (world box + imported part).
scene.export_node_stl("world", "world_subtree.stl", resolved=False)
