"""Generate analytical MiniGeom JSON without requiring OpenMC or mesh booleans."""
from minigeom import Box, CSGCompiler, Cylinder, Scene, Sphere

scene = Scene("csg_demo")
world = scene.define("world", Box(100, 100, 100))
moderator = scene.define("moderator", Cylinder(20, 60))
source = scene.define("source", Sphere(2))
world.place(moderator, priority=10)
moderator.place(source, at=(0, 0, 0), priority=20)

# A cylinder can protrude beyond its parent, but remains inside the world.
channel = scene.define("channel", Cylinder(3, 80))
moderator.place_protruding(channel, at=(0, 0, 0), overlap_priority=100)

model = CSGCompiler(boundary_type="vacuum").compile(scene)
model.export_json("csg_demo.json")
print(f"Wrote csg_demo.json ({len(model.surfaces)} surfaces, {len(model.cells)} cells)")
