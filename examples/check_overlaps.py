"""Run the deterministic sampled overlap/partition diagnostic."""
from minigeom import Box, Scene, Sphere

scene = Scene("overlap_diagnostic")
world = scene.define("world", Box(20, 20, 20), material="air")
left = scene.define("left", Sphere(4), material="steel")
right = scene.define("right", Sphere(4), material="lead")
world.place(left, at=(-2, 0, 0), priority=1)
world.place(right, at=(2, 0, 0), priority=10)

report = scene.check_overlaps(samples=12000)
print(report.summary())
report.export_json("overlap_report.json")

if report.has_partition_issues:
    raise SystemExit("Unresolved overlap/partition issue detected")
