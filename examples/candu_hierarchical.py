"""Build the TRIPOLI-5 CANDU 37-pin assembly with MiniGeom.

Geometry dimensions and pin-ring arrangement follow the official TRIPOLI-5
CANDU example (Pounders et al., 2011 reference configuration).
https://tripoli5.asnr.dev/documentation/latest-release/examples/geometries/candu.html

This script creates the analytical MiniGeom scene and can export it directly
to a TRIPOLI-5 AGORA ``.agora`` file when the TRIPOLI-5 Python API is installed.
All dimensions are in centimetres. The model is a single repeating assembly
with reflective boundary conditions on the six outer faces.
"""
import math
from minigeom import Box, Cylinder, Scene, Tripoli5Exporter

# Reference geometry parameters (cm)

# Fuel and cladding radii.
R_FUEL = 0.6103
R_CLAD = 0.6522

# Pressure tube and calandria radii.
PRESSURE_TUBE_INNER = 5.16890
PRESSURE_TUBE_OUTER = 5.60320
CALANDRIA_INNER = 6.44780
CALANDRIA_OUTER = 6.58750

# Fuel-pin ring radii and azimuthal configurations.
RING_RADII = (0.0, 1.4880, 2.8755, 4.3305)
PINS_PER_RING = (1, 6, 12, 18)
RING_ANGLE_OFFSETS_DEG = (0.0, 0.0, 15.0, 0.0)

# Repeating assembly pitch and axial height.
ASSEMBLY_PITCH = 28.5750
AXIAL_PITCH = 49.5300

# The moderator rings meet halfway between neighbouring pin-ring centre radii.
# These are geometric interfaces, not pin centre positions.
RING_INTERFACES = tuple(
    0.5 * (RING_RADII[i] + RING_RADII[i + 1])
    for i in range(len(RING_RADII) - 1)
)

# MiniGeom model construction
def _place_pin_ring(parent, pin_definition, ring_index: int) -> None:
    """Place one concentric ring of identical pins in a water-region volume."""
    radius = RING_RADII[ring_index]
    count = PINS_PER_RING[ring_index]
    angle_offset = RING_ANGLE_OFFSETS_DEG[ring_index]

    for pin_index in range(count):
        theta_deg = angle_offset + pin_index * 360.0 / count
        theta = math.radians(theta_deg)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        parent.place(
            pin_definition,
            at=(x, y, 0.0),
            priority=10,
            name=f"pin_ring{ring_index}_{pin_index:02d}",
        )

scene = Scene("candu_assembly")

# Outer repeating cell: the moderator outside the calandria tube.
assembly = scene.define(
    "assembly",
    Box(ASSEMBLY_PITCH, ASSEMBLY_PITCH, AXIAL_PITCH),
    material="heavy water",
)

# Nested radial shells. Each child occupies the inner part of its parent;
# the parent's residual region is therefore the corresponding annulus.
calandria = scene.define("calandria", Cylinder(CALANDRIA_OUTER, AXIAL_PITCH), material="zircaloy c")
filler = scene.define("filler", Cylinder(CALANDRIA_INNER, AXIAL_PITCH), material="filler")
pressure_tube = scene.define("pressure_tube", Cylinder(PRESSURE_TUBE_OUTER, AXIAL_PITCH), material="zircaloy pt")
bundle = scene.define("bundle", Cylinder(PRESSURE_TUBE_INNER, AXIAL_PITCH), material="heavy water")

assembly.place(calandria, priority=100, name="calandria_shell_domain")
calandria.place(filler, priority=100, name="filler_domain")
filler.place(pressure_tube, priority=100, name="pressure_tube_domain")
pressure_tube.place(bundle, priority=100, name="bundle_domain")

# Fuel pin universe: fuel occupies the inner cylinder; the residual inside
# the outer cladding cylinder is the zircaloy cladding.
pin = scene.define("fuel_pin", Cylinder(R_CLAD, AXIAL_PITCH), material="zircaloy")
fuel = scene.define("fuel", Cylinder(R_FUEL, AXIAL_PITCH), material="fuel")
pin.place(fuel, priority=100, name="fuel_region")

# The water rings are nested cylinders. Their residual volumes are the
# successive annuli; pin placements are siblings in the corresponding ring.
water_ring_0 = scene.define("water_ring_0", Cylinder(RING_INTERFACES[0], AXIAL_PITCH), material="heavy water")
water_ring_1 = scene.define("water_ring_1", Cylinder(RING_INTERFACES[1], AXIAL_PITCH), material="heavy water")
water_ring_2 = scene.define("water_ring_2", Cylinder(RING_INTERFACES[2], AXIAL_PITCH), material="heavy water")

bundle.place(water_ring_2, priority=100, name="inner_water_regions")
water_ring_2.place(water_ring_1, priority=100, name="water_ring_1_domain")
water_ring_1.place(water_ring_0, priority=100, name="water_ring_0_domain")

# Distribute all 37 pins across their corresponding radial moderator zones.
_place_pin_ring(water_ring_0, pin, 0)
_place_pin_ring(water_ring_1, pin, 1)
_place_pin_ring(water_ring_2, pin, 2)
_place_pin_ring(bundle, pin, 3)

scene.set_top(assembly)

# Direct AGORA export requires an installed TRIPOLI-5 Python API.
exporter = Tripoli5Exporter(
    scene,
    boundary_type="reflective",
    mode="hierarchy",
    optimize_repeated=True,
)
agora_path = exporter.export("candu_assembly.agora")
