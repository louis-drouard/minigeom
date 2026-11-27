# MiniGeom 0.2.0

MiniGeom is a backend-independent analytical geometry builder for Monte-Carlo transport geometries. It provides two complementary ways to describe a geometry:

1. **Implicit hierarchy**: define reusable volumes and place children inside parent volumes. MiniGeom resolves the resulting material regions from containment and priority rules.
2. **Explicit Boolean regions**: define cells directly with set operations on analytical primitives (`&`, `|`, `-`, `~`). MiniGeom compiles those expressions into the same analytical CSG representation.

Both descriptions feed the same analytical surface registry and the same OpenMC/TRIPOLI-5 exporters. Triangle meshes are a separate, optional representation used only for previews, STL import/export and approximate mesh booleans. They are never used to produce transport-code boundaries.

## Architecture

```text
                         USER MODEL
                 Scene / VolumeDef / Primitive
                              |
              +---------------+----------------+
              |                                |
       Implicit hierarchy                Explicit Boolean cells
       place / place_lattice             add_cell(region=...)
              |                                |
              +---------------+----------------+
                              |
                         CSGCompiler
                              |
         SurfaceRegistry / analytical surfaces / Region AST
                              |
                 +------------+------------+
                 |                         |
             OpenMC XML                TRIPOLI-5 AGORA

       Optional display/interchange path (independent):
       Primitive.mesh / MeshPrimitive / Trimesh / PyVista / STL
```

### Source modules

- `geometry.py`: analytical primitives, reusable definitions, placements and lattices.
- `region_expr.py`: user-facing Boolean expression tree. It contains no OpenMC, AGORA or Trimesh code.
- `scene.py`: scene graph, instance traversal and optional mesh visualization/export.
- `csg.py`: surface registry, low-level half-space/region AST and the common CSG compiler.
- `overlap.py`: sampled overlap and partition diagnostics.
- `openmc_exporter.py`: OpenMC XML, numeric material IDs and optional real `materials.xml`.
- `tripoli5_exporter.py`: TRIPOLI-5 AGORA geometry export.

## Installation

The analytical core requires only NumPy:

```bash
pip install -e .
```

### Optional mesh features

Trimesh is not a core dependency. Install it only when you need STL import/export or primitive preview meshes:

```bash
pip install -e ".[mesh]"
```

Install the Manifold backend for approximate mesh boolean operations (preview/STL overlap resolution):

```bash
pip install -e ".[mesh-boolean]"
```

Install the complete interactive visualization stack:

```bash
pip install -e ".[visualization]"
```

These mesh booleans are **not** used by the OpenMC or TRIPOLI-5 exporters. PyVista and Trimesh are optional and imported lazily only when a mesh/visualization method is called.

### Optional transport-code APIs

OpenMC is installed separately according to the official installation instructions for the user's platform. TRIPOLI-5 and its Python API are distributed through the vendor's supported channels. MiniGeom itself imports without either transport-code package installed; the corresponding exporter checks the dependency when instantiated.

- [OpenMC documentation](https://docs.openmc.org/en/stable/)
- [TRIPOLI-5 Python API](https://tripoli5.asnr.dev/documentation/latest-release/api/python-api.html)
- [TRIPOLI-5 AGORA API](https://tripoli5.asnr.dev/documentation/latest-release/api/agora-api.html)

## Quick start: implicit hierarchy

This is the concise ROOT-like construction style. A child occupies its footprint inside the parent; the parent's residual region keeps the parent's material. Sibling overlaps are resolved by priority (higher integer wins, then later insertion wins ties).

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

This style is convenient for nested mechanical assemblies, repeated definitions and rectangular lattices.

## Explicit Boolean regions

Use this style when the cell partition is easier to understand as direct set algebra than as a parent/child hierarchy. Boolean operators create a **lazy analytical expression**; they do not call Trimesh, Manifold or any mesh boolean routine.

| Python operator | Set operation | Meaning |
|---|---|---|
| `A & B` | Intersection | Points in both A and B |
| `A | B` | Union | Points in A or B (or both) |
| `A - B` | Difference | Points in A but not in B |
| `~A` | Complement | Points outside A |

Primitives can be used directly as operands. Transformations can be attached to a primitive/expression before it is added to a cell:

```python
from minigeom import Box, Cylinder, Scene, Sphere, union_all, CSGCompiler

scene = Scene("boolean_detector")
world = scene.define("world", Box(100, 100, 100))

outer = Cylinder(20, 60)
inner = Cylinder(15, 60)
shield_shell = outer - inner

# Two separated spheres form one disconnected CSG region/cell.
targets = union_all([
    Sphere(3).translated((-10, 0, 0)),
    Sphere(3).translated((+10, 0, 0)),
])

world.add_cell("shield", shield_shell, material="lead")
world.add_cell("targets", targets, material="tungsten", priority=10)

model = CSGCompiler().compile(scene)
model.export_json("boolean_detector.json")
```

`VolumeDef.add_cell(name, region, material=None, priority=0)` declares a named material cell in that universe. Its expression is evaluated in the owning volume's local coordinate system, transformed to world coordinates, and clipped to the owning volume's allowed domain. Explicit cells in one universe are arbitrated by `priority`, then declaration order (last declaration wins ties).

A universe must currently use **one partitioning style at a time**:

- implicit children/lattices via `place()` and `place_lattice()`, or
- explicit cells via `add_cell()`.

Mixing both styles in the same `VolumeDef` is rejected, because the two lists would otherwise need a shared priority/order policy. They can still be used at different levels of a scene hierarchy.

`~A` is mathematically unbounded. When used in an explicit cell, the result is clipped to the bounded domain of its owning `VolumeDef`; that owning volume must therefore have a finite analytical shape or be placed inside another finite domain.

### Expression helpers

`union_all([...])` and `intersection_all([...])` build unions/intersections from an iterable and reject empty input. These are useful for repeated geometry such as fuel pins:

```python
pin_holes = union_all(
    Cylinder(0.65, 50).translated((x, y, 0))
    for x, y in pin_positions
)

water_region = moderator_annulus - pin_holes
```

`RegionExpression.translated()` and `.rotated()` apply rigid transforms to every primitive leaf in an expression. Rotations use MiniGeom's XYZ Euler convention in degrees (`Rz @ Ry @ Rx`). Scaling and shear are deliberately not supported for analytical expression transforms.

## CANDU 37-pin examples

Two complete constructions of the same CANDU bundle are included in `examples/`:

### 1. Hierarchical construction

[`examples/candu_hierarchical.py`](examples/candu_hierarchical.py)

- Reuses one fuel-pin `VolumeDef` at 37 positions.
- Uses parent/child residual regions for fuel/cladding and the radial water shells.
- Builds the 1, 6, 12 and 18 pin rings with placements.
- Uses the hierarchy-preserving AGORA path (`mode="hierarchy"`).

Run the analytical point checks without requiring TRIPOLI-5:

```bash
python examples/candu_hierarchical.py --csg-only
```

Export AGORA with an installed TRIPOLI-5 Python API:

```bash
python examples/candu_hierarchical.py --output-dir out
```

### 2. Explicit Boolean-region construction

[`examples/candu_boolean_regions.py`](examples/candu_boolean_regions.py)

- Builds each fuel/cladding cell as a union of the corresponding 37 translated cylinders/annuli.
- Defines the four moderator rings as radial disks/annuli minus the appropriate pin footprints.
- Defines the pressure tube, filler, calandria and outer moderator with direct differences.
- Uses one assembly universe with 10 explicit material cells. The root `VolumeDef` has `material=None`, so no implicit residual cell is created; the 10 Boolean regions define the complete material partition.
- Exports through the flat CSG backend because the cells are already expressed as explicit regions.

Run its analytical checks and write the intermediate CSG JSON:

```bash
python examples/candu_boolean_regions.py --export-csg-json
```

Optional transport exports:

```bash
python examples/candu_boolean_regions.py --export-tripoli5
python examples/candu_boolean_regions.py --export-openmc
```

The two examples intentionally preserve the same physical arrangement and representative material-point checks. The hierarchy version produces 82 flattened cell instances; the explicit Boolean version produces 10 material cells by grouping the 37 fuel pellets and 37 cladding annuli into disconnected union regions. Both use the same 87 unique analytical surfaces. This is a difference in representation, not in the intended material distribution. See [`examples/README.md`](examples/README.md) for details.

## Analytical primitives

- `Sphere(radius)`
- `Cylinder(radius, height)` — finite cylinder centred at the local origin, axis along local Z.
- `Box(x, y, z)`
- `Cone(bottom_radius, top_radius, height)`
- `Ellipsoid((rx, ry, rz))`
- `Torus(major_radius, minor_radius, axis="z")`
- `Plane(normal, offset=0, sense=-1)`
- `Quadric(coefficients, sense=-1, bounded=False, bounds=None)`

The CSG compiler retains analytical surfaces and half-space expressions. It does not tessellate these primitives for transport export.

## Material tags and OpenMC export

A MiniGeom material is an optional symbolic string, not a physical material definition. See `materials_ids.json` for the tag-to-numeric-ID mapping produced by `OpenMCExporter`. `materials.xml` is written only when real `openmc.Material` objects are supplied and used by exported cells. Otherwise, non-void symbolic tags are still written as numeric references in `geometry.xml` and listed as external in the mapping JSON.

```python
from minigeom import OpenMCExporter

# The scene can contain symbolic tags such as "uo2" and "water".
exporter = OpenMCExporter(scene)
exporter.export_geometry("openmc_geometry")
```

For complete details, partial material binding and explicit material IDs, see the material section in the previous 0.7.x documentation history or the `OpenMCExporter` docstring.

Explicit Boolean cells automatically select the flat CSG export path in `mode="auto"`. Forcing `mode="hierarchy"` with explicit region cells raises a clear error; hierarchy and explicit-cell partitioning are not silently mixed.

## TRIPOLI-5 export

`Tripoli5Exporter` accepts both construction styles:

- implicit volume hierarchies can preserve AGORA universes and rectangular lattices;
- explicit Boolean cells are exported through the flat AGORA CSG path.

```python
from minigeom import Tripoli5Exporter

exporter = Tripoli5Exporter(scene, boundary_type="reflective", mode="auto")
exporter.export("geometry.agora")
```

TRIPOLI-5 requires every material leaf to carry a material-name string (or the exporter must receive `default_material`). MiniGeom does not create TRIPOLI-5 nuclear material definitions.

## Optional mesh workflow: preview and STL only

Meshes are a separate representation. They are useful for visual inspection, CAD interchange and checking a model visually, but they do not define the simulation geometry. Analytical exporters reject `MeshPrimitive` because a triangle mesh is not an exact analytical boundary.

Install the optional mesh dependency first:

```bash
pip install -e ".[mesh]"
```

Import an STL:

```python
scene = Scene("cad_import")
part = scene.import_stl(
    "cad_part", "cad_part.stl", scale=0.1,
    material="steel", require_watertight=True,
)
```

Export a placed instance or node:

```python
scene.export_instance_stl("world/cad_part#1", "part_world.stl", resolved=False)
scene.export_node_stl("world", "world_subtree.stl", resolved=False)
```

Export a mesh preview of the complete scene:

```python
scene.export_stl("scene_preview.stl", resolved=False)
scene.show(resolved=False)
```

`resolved=True` applies priority rules using approximate triangle-mesh booleans. It requires `pip install -e ".[mesh-boolean]"` and can introduce tessellation artifacts. It is never used by the OpenMC or TRIPOLI-5 exporters.

## Overlap diagnostics

```python
report = scene.check_overlaps(samples=20000)
print(report.summary())
report.export_json("overlap_report.json")
```

The checker uses deterministic low-discrepancy sampling. It is a diagnostic, not a formal proof; small overlaps can be missed. Explicit region-cell raw footprints are included in the pair checks.

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

The test suite covers implicit CSG, explicit Boolean regions, the two CANDU examples, STL workflows and mocked OpenMC/TRIPOLI-5 API object graphs. Mocked exporters do not replace a run with the installed transport-code release; validate representative XML/AGORA outputs with your actual installations before production use.

## Current limitations

- A universe cannot currently mix explicit `add_cell()` definitions with implicit child/lattice placements in the same `VolumeDef`.
- Only rectangular lattices are exposed by the MiniGeom API; hexagonal lattices are not implemented.
- The root of a rectangular lattice tile must be a `Box` matching the lattice pitch.
- STL imports are mesh-backed and cannot be exported as analytical OpenMC/TRIPOLI-5 surfaces without manual reconstruction.
- Mesh boolean results are approximate, preview-oriented tessellations.
- The overlap checker is sampling-based, not a formal mathematical proof.
- OpenMC torus axes must be aligned with a global coordinate axis after placement.
- The documented AGORA Python API does not currently expose a torus surface; TRIPOLI-5 export rejects torus regions.

## Lattice element substitutions (0.9.0)

A regular lattice can now use a default tile universe and sparse replacements
without flattening the native lattice. Indices use MiniGeom's natural `(i, j, k)`
convention: `i` increases along X, `j` along Y, and `k` along Z.

```python
assembly = scene.define("assembly", Box(17*pitch, 17*pitch, height), material="water")
lattice = assembly.place_lattice(
    fuel_tile, shape=(17, 17, 1), pitch=(pitch, pitch, height),
    lower_left=(-8.5*pitch, -8.5*pitch, -height/2), name="fuel_lattice"
)
lattice[5, 2, 0] = guide_tile
lattice[8, 2, 0] = guide_tile

# Read the element definition, or restore the default tile.
assert lattice[5, 2, 0] is guide_tile
lattice.clear_element(5, 2, 0)
assert lattice[5, 2, 0] is fuel_tile
```

The replacements are represented as native per-element universe fills in OpenMC
`RectLattice.universes` and AGORA `RectLattice`, not as priority-overlaid volumes.
See [`examples/lattice_with_replacements.py`](examples/lattice_with_replacements.py).
