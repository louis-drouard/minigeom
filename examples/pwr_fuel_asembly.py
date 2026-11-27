from minigeom import Box, Cylinder, Scene, Tripoli5Exporter

pincell_pitch = 1.262082
nb_pincells = 17
pitch = pincell_pitch * nb_pincells
height = 427.0
water_blade_thickness = 0.07792594
total_size = pitch+water_blade_thickness*2
scene = Scene()

water_background = scene.define("assembly",Box(total_size, total_size, height), 'water')

fuel_pincell = scene.define("fuel_pincell", Box(pincell_pitch, pincell_pitch, height), "water")
fuel = scene.define("fuel", Cylinder(0.41266, height), "fuel")
clad_ir = scene.define("clad_ir", Cylinder(0.41690, height), "void")
clad_or = scene.define("clad_or", Cylinder(0.47436, height), "clad")
clad_ir.place(fuel)
clad_or.place(clad_ir)
fuel_pincell.place(clad_or)

gt_pincell = scene.define("gt_pincell", Box(pincell_pitch, pincell_pitch, height), "water")
gtclad_ir = scene.define("gtclad_ir", Cylinder(0.56343, height), "water")
gtclad_or = scene.define("gtclad_or" , Cylinder(0.6035, height), "clad")
gtclad_or.place(gtclad_ir)
gt_pincell.place(gtclad_or)

lattice = water_background.place_lattice(
    fuel_pincell,
    shape=(nb_pincells, nb_pincells, 1),
    pitch=(pincell_pitch, pincell_pitch, height),
    lower_left=(-pitch/2, -pitch/2, -height/2),
    priority=10
)

# Set guide tubes
lattice[5, 2, 0] = gt_pincell
lattice[8, 2, 0] = gt_pincell
lattice[11, 2, 0] = gt_pincell
lattice[3, 3, 0] = gt_pincell
lattice[13, 3, 0] = gt_pincell
lattice[2, 5, 0] = gt_pincell
lattice[5, 5, 0] = gt_pincell
lattice[8, 5, 0] = gt_pincell
lattice[11, 5, 0] = gt_pincell
lattice[14, 5, 0] = gt_pincell
lattice[2, 8, 0] = gt_pincell
lattice[5, 8, 0] = gt_pincell
lattice[8, 8, 0] = gt_pincell
lattice[11, 8, 0] = gt_pincell
lattice[14, 8, 0] = gt_pincell
lattice[2, 11, 0] = gt_pincell
lattice[5, 11, 0] = gt_pincell
lattice[8, 11, 0] = gt_pincell
lattice[11, 11, 0] = gt_pincell
lattice[14, 11, 0] = gt_pincell
lattice[3, 13, 0] = gt_pincell
lattice[13, 13, 0] = gt_pincell
lattice[5, 14, 0] = gt_pincell
lattice[8, 14, 0] = gt_pincell
lattice[11, 14, 0] = gt_pincell

scene.set_top(water_background)

exporter = Tripoli5Exporter(scene, boundary_type="reflective")
agora_path = exporter.export("candu_assembly.agora")
