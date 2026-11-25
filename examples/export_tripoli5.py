"""Export an analytical scene to the TRIPOLI-5 AGORA geometry format."""
from minigeom import Box, Cylinder, Scene, Sphere, Tripoli5Exporter

scene = Scene("tripoli_detector")
world = scene.define("world", Box(100, 100, 100), material="AIR")
shield = scene.define("shield", Cylinder(20, 60), material="LEAD")
target = scene.define("target", Sphere(3), material="TUNGSTEN")
world.place(shield, priority=10)
shield.place(target, priority=20)

exporter = Tripoli5Exporter(scene, boundary_type="leakage", mode="auto")
exporter.export("tripoli_detector.agora")
