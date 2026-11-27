
from minigeom import (
    Scene, Box, Cylinder, OpenMCExporter
)

H = 365.0

N_PIN = 17
PIN_PITCH = 1.26

ASSEMBLY_PITCH = 21.60

N_ASSEMBLY_X = 3
N_ASSEMBLY_Y = 3


scene = Scene("PWR_core")

fuel = scene.define(
    "fuel",
    Cylinder(outer_radius=0.4095, height=H),
    material="uo2"
)

gap = scene.define(
    "gap",
    Cylinder(outer_radius=0.418, height=H),
    material="helium"
)

clad = scene.define(
    "cladding",
    Cylinder(outer_radius=0.475, height=H),
    material="zircaloy"
)

# Hiérarchie radiale :
# Cladding > Gap > Fuel
gap.place(fuel, priority=20)
clad.place(gap, priority=10)

# CELLULE ELEMENTAIRE DU RESEAU DE CRAYONS
pin_tile = scene.define(
    "pin_tile",
    Box(PIN_PITCH, PIN_PITCH, H),
    material="water"
)

pin_tile.place(clad, priority=10)

# DEFINITION D'UN ASSEMBLAGE
assembly = scene.define(
    "assembly",
    Box(ASSEMBLY_PITCH, ASSEMBLY_PITCH, H),
    material="water"
)

# Réseau 17 x 17 de crayons
assembly.place_lattice(
    pin_tile,
    shape=(N_PIN, N_PIN, 1),
    pitch=(PIN_PITCH, PIN_PITCH, H),
    lower_left=(
        -N_PIN * PIN_PITCH / 2,
        -N_PIN * PIN_PITCH / 2,
        -H / 2
    ),
    priority=10,
    name="fuel_pins_17x17"
)
scene.set_top(assembly)

# DEFINITION DU COEUR
core = scene.define(
    "core",
    Box(
        N_ASSEMBLY_X * ASSEMBLY_PITCH,
        N_ASSEMBLY_Y * ASSEMBLY_PITCH,
        H
    ),
    material="water"
)

# Réseau externe d'assemblages
core.place_lattice(
    assembly,
    shape=(N_ASSEMBLY_X, N_ASSEMBLY_Y, 1),
    pitch=(ASSEMBLY_PITCH, ASSEMBLY_PITCH, H),
    lower_left=(
        -N_ASSEMBLY_X * ASSEMBLY_PITCH / 2,
        -N_ASSEMBLY_Y * ASSEMBLY_PITCH / 2,
        -H / 2
    ),
    priority=10,
    name="assembly_lattice"
)

scene.set_top(core)

# EXPORT OPENMC
import openmc
moderator = openmc.Material(name="moderator")
moderator.set_density("g/cm3", 1.0)
moderator.add_nuclide("H1", 2.0)
moderator.add_nuclide("O16", 1.0)
fuel_mat = openmc.Material(name="fuel")
fuel_mat.set_density("g/cm3", 10.0)
fuel_mat.add_nuclide("U235", 0.04)
fuel_mat.add_nuclide("U238", 0.96)
zr_mat = openmc.Material(name="zircaloy")
zr_mat.set_density("g/cm3", 6.35)
zr_mat.add_nuclide("Zr95", 1.0)
gap_mat = openmc.Material(name="helium")
gap_mat.set_density("g/cm3", 0.01)
gap_mat.add_nuclide("He4", 1.0)

exporter = OpenMCExporter(
    scene,
    materials_map={'uo2': fuel_mat, 'water': moderator, 'zircaloy': zr_mat, 'helium': gap_mat},
    mode="hierarchy",
    optimize_repeated=True,
    use_lattices=True
)
exporter.export_geometry("openmc_pwr")

print("Export terminé")
print("Universes :", len(exporter.universes))
print("Lattices :", len(exporter.lattices))
print("Cells :", len(exporter.cells))
