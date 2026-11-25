"""Example: build a detector with optional explicit OpenMC material binding.

Run after installing OpenMC. For geometry-only use, omit materials_map and
call OpenMCExporter(scene).export_geometry(...).
"""
import openmc

from minigeom import Box, Cylinder, Scene, Sphere, OpenMCExporter

scene = Scene("detector")
world = scene.define("world", Box(200.0, 200.0, 200.0), material="air")
shield = scene.define("shield", Cylinder(radius=45.0, height=120.0), material="lead")
target = scene.define("target", Sphere(radius=12.0), material="tungsten")
pipe = scene.define("pipe", Cylinder(radius=4.0, height=80.0), material="steel")

world.place(shield, at=(0.0, 0.0, 0.0), priority=10)
world.place(target, at=(0.0, 0.0, 0.0), priority=20)
world.place(pipe, at=(0.0, 0.0, 35.0), rotate=(0.0, 25.0, 0.0), priority=30)
scene.set_top(world)

# Important: these are illustrative material definitions. Replace densities
# and nuclide compositions with the actual material data for your model.
air = openmc.Material(name="air")
air.set_density("g/cm3", 0.001205)
air.add_element("N", 0.78, percent_type="ao")
air.add_element("O", 0.21, percent_type="ao")
air.add_element("Ar", 0.01, percent_type="ao")

lead = openmc.Material(name="lead")
lead.set_density("g/cm3", 11.34)
lead.add_element("Pb", 1.0)

tungsten = openmc.Material(name="tungsten")
tungsten.set_density("g/cm3", 19.3)
tungsten.add_element("W", 1.0)

steel = openmc.Material(name="steel")
steel.set_density("g/cm3", 7.9)
steel.add_element("Fe", 0.70)
steel.add_element("Cr", 0.19)
steel.add_element("Ni", 0.11)

exporter = OpenMCExporter(scene, {
    "air": air,
    "lead": lead,
    "tungsten": tungsten,
    "steel": steel,
}, boundary_type="vacuum")
exporter.export("openmc_detector")
print("OpenMC files written to openmc_detector/")
