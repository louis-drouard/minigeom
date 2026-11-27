"""Export an OpenMC geometry deck with symbolic material IDs but no materials.xml."""
from minigeom import Box, Cylinder, OpenMCExporter, Scene, Sphere

scene = Scene("geometry_only_detector")
world = scene.define("world", Box(100, 100, 100), material="air")
shield = scene.define("shield", Cylinder(outer_radius=20, height=60), material="lead")
target = scene.define("target", Sphere(3), material="tungsten")
world.place(shield, priority=10)
shield.place(target, priority=20)

exporter = OpenMCExporter(scene)
exporter.export_geometry("openmc_geometry_only")
print("Exported geometry.xml and materials_ids.json; materials.xml was not created.")
