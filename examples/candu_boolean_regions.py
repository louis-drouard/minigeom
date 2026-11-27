"""CANDU 37-pin bundle expressed with explicit Boolean regions.

This is the declarative counterpart of ``candu_hierarchical.py``. It uses the
same reference dimensions, pin coordinates, region materials and reflective
outer boundary, but creates the material cells from Boolean expressions rather
than inferring annular residuals from nested parent/child volumes.

All dimensions are in centimetres. The example is intentionally verbose around
region construction: the comments explain which set operation creates each
physical cell and why the expressions are clipped to the assembly universe.

The resulting explicit-cell universe is exported through MiniGeom's flat CSG
backend. The flat backend preserves the Boolean surfaces/cells directly and is
supported by both the OpenMC and TRIPOLI-5 exporters.
"""
import argparse
import importlib.util
import math
from pathlib import Path

from minigeom import (
    Box,
    CSGCompiler,
    Cylinder,
    Scene,
    Tripoli5Exporter,
    OpenMCExporter,
    union_all,
)


# ---------------------------------------------------------------------------
# Reference geometry parameters (cm)
# ---------------------------------------------------------------------------

R_FUEL = 0.6103
R_CLAD = 0.6522

PRESSURE_TUBE_INNER = 5.16890
PRESSURE_TUBE_OUTER = 5.60320
CALANDRIA_INNER = 6.44780
CALANDRIA_OUTER = 6.58750

RING_RADII = (0.0, 1.4880, 2.8755, 4.3305)
PINS_PER_RING = (1, 6, 12, 18)
RING_ANGLE_OFFSETS_DEG = (0.0, 0.0, 15.0, 0.0)

ASSEMBLY_PITCH = 28.5750
AXIAL_PITCH = 49.5300

# The moderator rings meet halfway between neighbouring pin-ring centre radii.
# These are geometric interfaces, not pin centre positions.
RING_INTERFACES = tuple(
    0.5 * (RING_RADII[index] + RING_RADII[index + 1])
    for index in range(3)
)

# Coordinate generation
def pin_positions_by_ring() -> dict[int, list[tuple[float, float]]]:
    """Return the centre coordinates for the 1+6+12+18 pin arrangement."""
    result: dict[int, list[tuple[float, float]]] = {}
    for ring_index, (radius, count, offset_deg) in enumerate(
        zip(RING_RADII, PINS_PER_RING, RING_ANGLE_OFFSETS_DEG)
    ):
        points = []
        for pin_index in range(count):
            angle = math.radians(offset_deg + pin_index * 360.0 / count)
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
        result[ring_index] = points
    return result

def _translated_pin_shapes(radius: float, positions):
    """Create one translated finite-cylinder expression for every pin centre.

    ``Cylinder`` is a finite analytical solid in MiniGeom. ``translated`` does
    not generate a mesh: it attaches a rigid-transform matrix to the expression
    leaf. The CSG compiler later transforms the cylinder's side and cap surfaces
    into the assembly coordinate system.
    """
    local_cylinder = Cylinder(outer_radius=radius, height=AXIAL_PITCH)
    return [local_cylinder.translated((x, y, 0.0)) for x, y in positions]

# Boolean CANDU model
scene = Scene("candu_boolean_regions")

# The root Box is the finite universe boundary. Every explicit cell added
# below is interpreted in this local coordinate frame and clipped to this
# Box by CSGCompiler. No cell is allowed to extend beyond the assembly.
assembly = scene.define(
    "assembly",
    Box(ASSEMBLY_PITCH, ASSEMBLY_PITCH, AXIAL_PITCH),
    # None means that the explicit cells themselves define the complete
    # partition. No implicit catch-all residual cell is added.
    material=None,
)

positions = pin_positions_by_ring()

# Build the 37 outer cladding cylinders and the 37 fuel cylinders. Keeping
# these as separate primitive expressions lets the Boolean tree represent
# disconnected regions without creating 37 different VolumeDef objects.
outer_pin_shapes = {
    ring: _translated_pin_shapes(R_CLAD, ring_positions)
    for ring, ring_positions in positions.items()
}
fuel_pin_shapes = {
    ring: _translated_pin_shapes(R_FUEL, ring_positions)
    for ring, ring_positions in positions.items()
}

# -----------------------------------------------------------------------
# 1. Fuel and cladding cells
# -----------------------------------------------------------------------
# A fuel cell may consist of several disconnected cylindrical components.
# A union of all 37 translated fuel cylinders describes that as ONE CSG
# cell, carrying the single material tag "fuel".
all_fuel_regions = [
    region
    for ring_regions in fuel_pin_shapes.values()
    for region in ring_regions
]
assembly.add_cell(
    name="fuel",
    region=union_all(all_fuel_regions),
    material="fuel",
)

# Each cladding segment is the outer pin cylinder minus its inner fuel
# cylinder. The union of those 37 annular regions forms one cladding cell.
# This is the direct set-theoretic definition:
#     cladding = union_i (outer_pin_i \ fuel_i)
cladding_regions = []
for ring_index in range(4):
    for outer_shape, fuel_shape in zip(
        outer_pin_shapes[ring_index], fuel_pin_shapes[ring_index]
    ):
        cladding_regions.append(outer_shape - fuel_shape)

assembly.add_cell(
    name="cladding",
    region=union_all(cladding_regions),
    material="zircaloy",
)

# -----------------------------------------------------------------------
# 2. Four moderator rings inside the pressure tube
# -----------------------------------------------------------------------
# The water domains are radial disks/annuli. Each domain is then cut by the
# union of the outer cladding cylinders belonging to that pin ring. This
# ensures that the water cannot occupy fuel or cladding locations.
ring_domains = [
    # Ring 0: central disk, from radius 0 to interface 0.
    Cylinder(outer_radius=RING_INTERFACES[0], height=AXIAL_PITCH),
    # Ring 1: annulus between interfaces 0 and 1.
    Cylinder(outer_radius=RING_INTERFACES[1], height=AXIAL_PITCH)
    - Cylinder(outer_radius=RING_INTERFACES[0], height=AXIAL_PITCH),
    # Ring 2: annulus between interfaces 1 and 2.
    Cylinder(outer_radius=RING_INTERFACES[2], height=AXIAL_PITCH)
    - Cylinder(outer_radius=RING_INTERFACES[1], height=AXIAL_PITCH),
    # Ring 3: annulus between interface 2 and the pressure-tube inner wall.
    Cylinder(outer_radius=PRESSURE_TUBE_INNER, height=AXIAL_PITCH)
    - Cylinder(outer_radius=RING_INTERFACES[2], height=AXIAL_PITCH),
]

for ring_index, ring_domain in enumerate(ring_domains):
    pin_footprint = union_all(outer_pin_shapes[ring_index])
    # Set difference is crucial here: the water ring occupies its radial
    # annulus EXCEPT for the physical space occupied by the  outer pin shells.
    water_region = ring_domain - pin_footprint
    assembly.add_cell(
        name=f"heavy_water_ring_{ring_index}",
        region=water_region,
        material="heavy water",
    )

# -----------------------------------------------------------------------
# 3. Pressure tube, filler and calandria annuli
# -----------------------------------------------------------------------
# These are direct differences between concentric finite cylinders.
pressure_tube_outer = Cylinder(outer_radius=PRESSURE_TUBE_OUTER, height=AXIAL_PITCH)
pressure_tube_inner = Cylinder(outer_radius=PRESSURE_TUBE_INNER, height=AXIAL_PITCH)
calandria_outer = Cylinder(outer_radius=CALANDRIA_OUTER, height=AXIAL_PITCH)
calandria_inner = Cylinder(outer_radius=CALANDRIA_INNER, height=AXIAL_PITCH)

assembly.add_cell(
    name="pressure_tube",
    region=pressure_tube_outer - pressure_tube_inner,
    material="zircaloy pt",
)

assembly.add_cell(
    name="filler",
    region=calandria_inner - pressure_tube_outer,
    material="filler",
)

assembly.add_cell(
    name="calandria",
    region=calandria_outer - calandria_inner,
    material="zircaloy c",
)

# The square assembly universe is filled with heavy water outside the
# cylindrical calandria tube. Subtracting the cylinder from the Box leaves
# the four corner regions and the moderator around the tube.
assembly_box = Box(ASSEMBLY_PITCH, ASSEMBLY_PITCH, AXIAL_PITCH)
assembly.add_cell(
    name="outer_moderator",
    region=assembly_box - calandria_outer,
    material="heavy water",
)

scene.set_top(assembly)

if importlib.util.find_spec("tripoli5") is not None:
    try:
        exporter = Tripoli5Exporter(
            scene, boundary_type="reflective", mode="flat"
        )
        exporter.export("candu_boolean.agora")
    except ImportError:
        print("TRIPOLI-5 API unavailable: skipping AGORA export.")
else:
    print("TRIPOLI-5 not installed: skipping AGORA export.")

if importlib.util.find_spec("openmc") is not None:
    try:
        exporter = OpenMCExporter(
            scene, boundary_type="reflective", mode="flat"
        )
        exporter.export_geometry("openmc")
    except ImportError:
        print("OpenMC API unavailable: skipping OpenMC export.")
else:
    print("OpenMC not installed: skipping OpenMC export.")

