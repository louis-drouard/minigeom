"""Analytical examples for cones, ellipsoids, planes, tori and quadrics."""
from minigeom import Box, Cone, Ellipsoid, Plane, Quadric, Scene, Sphere, Torus, CSGCompiler

scene = Scene("analytic_primitives")
world = scene.define("world", Box(100, 100, 100), material="air")

cone = scene.define("cone", Cone(bottom_radius=8, top_radius=2, height=20), "steel")
ellipsoid = scene.define("ellipsoid", Ellipsoid((12, 5, 3)), "aluminium")
torus = scene.define("torus", Torus(major_radius=15, minor_radius=2, axis="z"), "copper")
halfspace = scene.define("halfspace", Plane((0, 0, 1), offset=0, sense=-1), "water")
quadric = scene.define(
    "quadric", Quadric((1, 1, 1, 0, 0, 0, 0, 0, 0, -9), sense=-1), "ceramic"
)

world.place(cone, at=(-25, 0, 0))
world.place(ellipsoid, at=(0, 0, 0), rotate=(0, 25, 0))
world.place(torus, at=(25, 0, 0))
# This half-space is clipped by the world Box.
world.place(halfspace, at=(0, 0, -20), priority=-1)
# Use a higher-priority local feature.
world.place(quadric, at=(0, 0, 0), priority=10)

model = CSGCompiler().compile(scene)
model.export_json("analytic_primitives.json")
print(f"Wrote analytic_primitives.json ({len(model.surfaces)} surfaces, {len(model.cells)} cells)")
