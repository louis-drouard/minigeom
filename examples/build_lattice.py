"""Build a 3-D regular pin lattice; OpenMCExporter emits one RectLattice."""
import openmc
from minigeom import Box, Cylinder, OpenMCExporter, Scene

scene = Scene("pin_assembly")
world = scene.define("world", Box(24, 24, 12), material="moderator")

# The tile root Box fills one lattice pitch completely. The moderator fills
# the tile outside its fuel rod.
pin = scene.define("pin", Box(2, 2, 10), material="moderator")
fuel = scene.define("fuel", Cylinder(outer_radius=0.55, height=9.8), material="fuel")
pin.place(fuel)

world.place_lattice(
    pin,
    shape=(9, 9, 1),
    pitch=(2, 2, 10),
    lower_left=(-9, -9, -5),
    name="fuel_pin_lattice",
)

moderator = openmc.Material(name="moderator")
moderator.set_density("g/cm3", 1.0)
moderator.add_nuclide("H1", 2.0)
moderator.add_nuclide("O16", 1.0)
fuel_mat = openmc.Material(name="fuel")
fuel_mat.set_density("g/cm3", 10.0)
fuel_mat.add_nuclide("U235", 0.04)
fuel_mat.add_nuclide("U238", 0.96)

exporter = OpenMCExporter(
    scene,
    materials_map={"moderator": moderator, "fuel": fuel_mat},
    mode="auto",
)
exporter.export("openmc_pin_assembly")
print("OpenMC universes:", len(exporter.universes))
print("OpenMC lattices:", len(exporter.lattices))
