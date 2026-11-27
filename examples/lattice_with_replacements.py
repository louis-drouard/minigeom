"""Example: 17x17 fuel lattice with sparse guide-tube replacements.

The lattice is kept native. The guide-tube positions do not become standalone
priority overlays; each is a per-element universe replacement on the lattice.
"""
from minigeom import Box, Cylinder, CSGCompiler, Scene


NX = NY = 17
PITCH = 1.26
HEIGHT = 40.0
LOWER_LEFT = (-NX * PITCH / 2, -NY * PITCH / 2, -HEIGHT / 2)


scene = Scene("17x17_with_guides")
# The enclosing assembly boundary is a finite box. The lattice tile's root box
# exactly matches one pitch, which is required by MiniGeom's lattice model.
assembly = scene.define("assembly", Box(NX * PITCH, NY * PITCH, HEIGHT), material="water")

# Standard fuel tile: moderator occupies the tile except where the cladding and
# fuel are placed. This universe will be repeated in the majority of lattice sites.
fuel_tile = scene.define("fuel_tile", Box(PITCH, PITCH, HEIGHT), material="water")
clad = scene.define("cladding", Cylinder(outer_radius=0.46, height=HEIGHT), material="zircaloy")
fuel = scene.define("fuel", Cylinder(outer_radius=0.40, height=HEIGHT), material="fuel")
clad.place(fuel, priority=100)
fuel_tile.place(clad, priority=10)

# Guide-tube tile: the same moderator-filled tile shape, but with an annular
# structural tube and a coolant interior. This universe is used only at selected
# lattice coordinates.
guide_tile = scene.define("guide_tile", Box(PITCH, PITCH, HEIGHT), material="water")
guide_wall = scene.define("guide_wall", Cylinder(outer_radius=0.58, height=HEIGHT), material="zircaloy")
guide_inner = scene.define("guide_coolant", Cylinder(outer_radius=0.50, height=HEIGHT), material="water")
guide_wall.place(guide_inner, priority=100)
guide_tile.place(guide_wall, priority=10)

# Construct the native 17x17 lattice using the fuel universe as its default.
# Indices are in MiniGeom's natural (i,j,k) order, with i->X, j->Y, k->Z.
lattice = assembly.place_lattice(
    fuel_tile,
    shape=(NX, NY, 1),
    pitch=(PITCH, PITCH, HEIGHT),
    lower_left=LOWER_LEFT,
    name="fuel_lattice",
)


# Replace a few default fuel-pin positions with guide-tube universes.
# The tuple indices identify the tile, not a global coordinate or offset.
for index in ((5, 2, 0), (8, 2, 0), (5, 8, 0), (8, 8, 0)):
    lattice[index] = guide_tile


# A later assignment overwrites the previous replacement at that position.
# lattice[5, 2, 0] = instrument_tube_tile
# Restore a site to the default fuel tile with lattice.clear_element(...).

scene.set_top(assembly)
model = CSGCompiler().compile(scene)

expanded = list(scene.iter_instances())
replaced = [inst for inst in expanded if inst.lattice_index in lattice.replacements]
print(f"Analytical surfaces: {len(model.surfaces)}")
print(f"Expanded instances: {len(expanded)}")
print(f"Lattice replacements: {len(lattice.replacements)}")
print("Replacement instance labels:")
for inst in replaced:
    print(f"  {inst.label}: {inst.volume_def.name}")
