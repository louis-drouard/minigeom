# MiniGeom 0.1.0

MiniGeom is a lightweight analytical CSG geometry builder. It provides a reusable volume-definition graph, placements, priority-based overlap resolution, rectangular lattices, analytical surfaces, mesh import/export, and optional exporters for OpenMC and TRIPOLI-5.

The central architecture is backend-independent:

- `geometry.py`: primitive definitions, placements, and lattice definitions.
- `scene.py`: scene graph, instance traversal, STL import/export, and previews.
- `csg.py`: analytical surfaces, region expressions, and priority resolution.
- `overlap.py`: sampled overlap and partition diagnostics.
- `openmc_exporter.py`: OpenMC XML export and symbolic material-ID binding.
- `tripoli5_exporter.py`: TRIPOLI-5 AGORA geometry export.

## Installation

```bash
pip install -e .
pip install pytest
```

OpenMC is required only for OpenMC XML export. Install it using the official OpenMC installation instructions.

TRIPOLI-5 is an optional vendor-distributed dependency. Install the TRIPOLI-5 Python API using the instructions provided with your TRIPOLI-5 distribution. MiniGeom imports without TRIPOLI-5 installed; the dependency is checked only when `Tripoli5Exporter` is instantiated.

For STL boolean resolution and PyVista previews:

```bash
pip install manifold3d pyvista
```

## Quick start

```python
from minigeom import Box, Cylinder, Scene, Sphere, CSGCompiler

scene = Scene("detector")
world = scene.define("world", Box(100, 100, 100), material="air")
shield = scene.define("shield", Cylinder(20, 60), material="lead")
target = scene.define("target", Sphere(3), material="tungsten")

world.place(shield, priority=10)
shield.place(target, priority=20)
scene.set_top(world)

csg = CSGCompiler().compile(scene)
csg.export_json("detector_csg.json")
```

Material names in MiniGeom are symbolic tags. MiniGeom does not define nuclear composition, density, temperature, or cross-section data.

## Analytical primitives

The current analytical primitives are:

- `Sphere(radius)`
- `Cylinder(radius, height)`
- `Box(x, y, z)`
- `Cone(bottom_radius, top_radius, height)`
- `Ellipsoid((rx, ry, rz))`
- `Torus(major_radius, minor_radius, axis="z")`
- `Plane(normal, offset=0, sense=-1)`
- `Quadric(coefficients, sense=-1, bounded=False, bounds=None)`

The CSG compiler preserves analytical surfaces. Meshes are not used as a substitute for analytical boundaries.

`Plane` and `Quadric` can be used as clipped child regions. They cannot be used as the scene root unless the root is explicitly bounded. For a generic quadric, provide `bounded=True` and an explicit AABB with `bounds=(lower, upper)`; MiniGeom intersects the analytical region with that AABB.

OpenMC and TRIPOLI-5 do not support every mathematical surface in the same way. The backend-specific limitations are listed below.

## Material tags and OpenMC material IDs

A `material` argument is optional and accepts either `None` or a symbolic string:

```python
world = scene.define("world", Box(100, 100, 100), material="water")
fuel = scene.define("fuel", Cylinder(0.4, 350), material="uo2")
```

The tag is kept in the CSG model. It is not a material definition.

### OpenMC export without `openmc.Material` objects

Every non-void tag receives an OpenMC material ID. By default, IDs are allocated deterministically from the sorted set of tags, starting at 1 and skipping IDs already assigned to supplied materials. The exporter writes:

```text
openmc_geometry/
    geometry.xml
    materials_ids.json
    minigeom_csg.json
    manifest.json
```

No `materials.xml` is written unless real `openmc.Material` objects are supplied and used by the scene.

```python
from minigeom import OpenMCExporter

exporter = OpenMCExporter(scene)
exporter.export_geometry("openmc_geometry")
```

Example `materials_ids.json`:

```json
{
  "schema_version": 1,
  "scene": "detector",
  "materials": {
    "uo2": {"id": 1, "source": "external"},
    "water": {"id": 2, "source": "external"}
  }
}
```

`geometry.xml` contains numeric material references. The JSON file is the bridge between the MiniGeom names and those IDs. This geometry file is not a complete transport model until the corresponding materials are defined externally in OpenMC.

For an existing OpenMC project or stable IDs shared across several geometry files, pass an explicit mapping:

```python
exporter = OpenMCExporter(
    scene,
    material_ids={"uo2": 101, "water": 102},
)
exporter.export_geometry("openmc_geometry")
```

The exporter does not instantiate fake `openmc.Material` objects to reserve IDs. It first creates unfilled cells in the in-memory OpenMC geometry and then patches the exported XML cell material attributes with the symbolic IDs. This avoids polluting OpenMC's process-wide material-ID allocator. As a consequence, `export()` is the authoritative path for external symbolic tags; an in-memory object returned by `build()` cannot itself carry a tag without a Material object.

### OpenMC export with real material objects

Pass `materials_map` only for tags that should be backed by actual `openmc.Material` objects:

```python
import openmc
from minigeom import OpenMCExporter

uo2 = openmc.Material(name="uo2")
uo2.set_density("g/cm3", 10.4)
uo2.add_element("U", 1.0, enrichment=4.5)
uo2.add_element("O", 2.0)

water = openmc.Material(name="water")
water.set_density("g/cm3", 1.0)
water.add_nuclide("H1", 2.0)
water.add_nuclide("O16", 1.0)

exporter = OpenMCExporter(
    scene,
    materials_map={"uo2": uo2, "water": water},
)
exporter.export("openmc_model")
```

In this mode:

- The supplied Material object's own ID is used in `geometry.xml`.
- The material IDs are recorded in `materials_ids.json`.
- `materials.xml` is written only if at least one real `openmc.Material` is actually referenced by an exported cell and `export_materials=True`.
- A symbolic tag without a corresponding Material object still receives an external numeric ID and is listed as `source: "external"` in `materials_ids.json`.
- `materials_map` may therefore be partial. It is not an all-or-nothing binding table.

If the material XML is managed by another part of the application, use:

```python
exporter.export_geometry("openmc_geometry")
```

This keeps numeric references in `geometry.xml` but never writes `materials.xml`.

The special tags `None`, `"void"`, `"vacuum"`, and `"none"` are treated as intentionally unfilled OpenMC cells and do not receive a material ID.

## STL import and export

STL is supported as a mesh interchange and preview format. STL does not reliably encode physical units, so always provide a scale when the file unit differs from the MiniGeom scene unit.

### Import an STL volume into a scene

```python
scene = Scene("cad_import")
world = scene.define("world", Box(100, 100, 100), material="air")

cad_part = scene.import_stl(
    "cad_part",
    "input/cad_part.stl",
    material="steel",
    scale=0.1,                 # for example, mm to cm
    require_watertight=True,
)
world.place(cad_part, at=(10, 0, 0))
```

`Scene.import_stl()` (also available as `Scene.load_stl()`) returns a `VolumeDef` containing a `MeshPrimitive`. The mesh can be placed, previewed, and exported again as STL.

Set `require_watertight=False` only when importing an intentionally open surface. An open surface is not a closed solid and should not be treated as a watertight volume in downstream boolean operations.

### Export a definition, instance, node, or whole scene

Export a definition in its local coordinates:

```python
cad_part.export_stl("cad_part_local.stl")
```

Export one placed instance in world coordinates:

```python
scene.export_instance_stl(
    "world/cad_part#1",
    "cad_part_world.stl",
    resolved=False,
)
```

Export a node and all descendants as one STL:

```python
scene.export_node_stl("world", "world_subtree.stl", resolved=False)
```

Export a particular reused definition occurrence:

```python
scene.export_volume_stl(
    cad_part,
    "cad_part_occurrence.stl",
    instance_path="world/cad_part#1",
    resolved=False,
)
```

Export the whole scene:

```python
scene.export_stl("scene.stl", resolved=False)
```

Use `resolved=True` to apply the scene's overlap-priority rules to the preview meshes. This is a tessellated mesh boolean operation, requires `manifold3d`, and is approximate relative to the analytical CSG model. Use `resolved=False` for raw transformed meshes without boolean clipping.

A `MeshPrimitive` is deliberately not accepted by the analytical OpenMC or TRIPOLI-5 exporters. STL import does not convert a mesh into an exact analytical primitive. For transport exports, use supported analytical primitives or manually approximate the imported shape with analytical CSG.

## TRIPOLI-5 export

MiniGeom provides a separate `Tripoli5Exporter` backend. It builds TRIPOLI-5 AGORA surfaces, regions, volumes, composite volumes, universes, and rectangular lattices. Material tags are passed to AGORA as strings; MiniGeom does not generate a TRIPOLI-5 material definition file.

```python
from minigeom import Tripoli5Exporter

exporter = Tripoli5Exporter(
    scene,
    boundary_type="leakage",
    mode="auto",
)
exporter.export("tripoli_geometry.agora")
```

The `.agora` file is the serialized AGORA geometry. The exporter also writes a `tripoli_geometry.agora.json` manifest next to it.

`Tripoli5Exporter` requires the TRIPOLI-5 Python API supplied with TRIPOLI-5. It is intentionally not installed as a normal PyPI extra because the package distribution is vendor-managed; see the [TRIPOLI-5 Python API documentation](https://tripoli5.asnr.dev/documentation/latest-release/api/python-api.html) and [AGORA API documentation](https://tripoli5.asnr.dev/documentation/latest-release/api/agora-api.html).

The current backend:

- Preserves nested universes and rectangular lattices where the scene is compatible.
- Passes symbolic material tags directly to AGORA `Volume` objects.
- Requires every leaf volume to have a material tag, or an explicit `default_material` on the exporter.
- Supports analytical planes, spheres, and general quadrics. Cylinders, cones, and ellipsoids are represented through quadric surfaces by the CSG compiler.
- Does not export `MeshPrimitive` objects as CSG.
- Does not currently export tori because the documented AGORA surface API does not expose a torus surface.
- Uses `mode="auto"` to fall back to flattened CSG for protruding placements or logical nodes that cannot be represented by the hierarchical backend.

Example with an untagged volume:

```python
exporter = Tripoli5Exporter(scene, default_material="WATER")
exporter.export("tripoli_geometry.agora")
```

AGORA applies affine transformations as a translation followed by a rotation. MiniGeom applies `R @ point + translation`; the exporter therefore pre-rotates the translation vector so that the resulting placement is consistent with MiniGeom's transform convention.

## Rectangular lattices and repeated universes

`VolumeDef.place_lattice()` creates a regular 3-D rectangular lattice:

```python
assembly.place_lattice(
    pin_tile,
    shape=(17, 17, 1),
    pitch=(1.26, 1.26, 365.0),
    lower_left=(-10.71, -10.71, -182.5),
    name="pins_17x17",
)
```

The lattice tile root must be a `Box` exactly matching the pitch. OpenMC and TRIPOLI-5 exporters preserve compatible lattices as native backend lattice objects rather than expanding every element into separate exported volumes. The analytical CSG compiler and sampled overlap checker expand lattice instances for point queries and diagnostics.

## Overlap diagnostics

```python
report = scene.check_overlaps(samples=20000)
print(report.summary())
report.export_json("overlap_report.json")

if report.has_partition_issues:
    raise RuntimeError("The resolved CSG partition contains an issue")
```

The checker uses deterministic low-discrepancy sampling over shared AABBs. Small overlaps can be missed, so this is a diagnostic tool rather than a formal proof of non-overlap. For critical models, add targeted point checks and validate the exported geometry with the target transport code.

## Tests

```bash
python -m pytest -q
```

The suite tests analytical CSG, STL I/O, OpenMC XML structure (with a small API double), and TRIPOLI-5 AGORA structure (with a small API double). The doubles validate MiniGeom's generated object graphs and serialization calls; they do not replace validation with installed OpenMC and TRIPOLI-5 releases.

## Known limitations

- Only rectangular lattices are exposed by the MiniGeom API; hexagonal lattices are not implemented yet.
- The lattice tile root must exactly match the lattice pitch and be a `Box`.
- STL imports are mesh-backed, not analytical CSG; imported meshes cannot be transported directly by OpenMC or TRIPOLI-5 through these exporters.
- The overlap checker is sampled and is not a formal proof.
- OpenMC tori must have an axis parallel to a global coordinate axis after placement.
- AGORA does not currently expose a torus surface in the documented Python API; MiniGeom's TRIPOLI-5 exporter rejects torus regions.
- The TRIPOLI-5 exporter has been tested with an API double in this build environment. Validate a representative `.agora` file with the actual TRIPOLI-5 installation before production use.
