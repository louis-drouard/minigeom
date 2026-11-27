# MiniGeom examples

All dimensions in the CANDU scripts are in centimetres. Both scripts describe
the same reference 37-pin assembly (1 + 6 + 12 + 18 pins), with the same
material layout and reflective outer boundary. They differ only in how the
user describes the partition of space.

## `candu_hierarchical.py` — implicit parent/child hierarchy

This is the more compact, ROOT-like style:

- define a pin once and place it at 37 coordinates;
- the pin's fuel child takes the inner cylindrical region;
- the parent pin retains the remaining annulus as cladding;
- nested radial containers create the four moderator regions;
- child placements and priorities define the material partition.

Run without OpenMC or TRIPOLI-5 installed:

```bash
python examples/candu_hierarchical.py --csg-only
```

Run the AGORA export with the TRIPOLI-5 Python API installed:

```bash
python examples/candu_hierarchical.py --output-dir output/hierarchical
```

## `candu_boolean_regions.py` — explicit set algebra

This version describes each material cell directly with analytical set
operations:

```python
cladding = union_all(
    outer_pin_i - fuel_pin_i
    for each pin i
)

water_ring_i = radial_ring_i - union_all(cladding_pin_i)
```

The script combines disconnected fuel pellets into one fuel cell and all
cladding annuli into one cladding cell. It then defines the four heavy-water
rings and the pressure-tube, filler, calandria and outer-moderator regions.
The root `VolumeDef` supplies a finite clipping domain but has no material tag,
so no implicit catch-all cell is added. The 10 explicit regions define the
intended material partition.

This is the preferred example when the geometric partition itself is easier
to understand as equations than as a hierarchy of nested volumes.

Run analytical point checks and write the intermediate CSG JSON:

```bash
python examples/candu_boolean_regions.py --export-csg-json
```

Export to TRIPOLI-5 AGORA or OpenMC XML with the corresponding API installed:

```bash
python examples/candu_boolean_regions.py --export-tripoli5
python examples/candu_boolean_regions.py --export-openmc
```

## What the comparison shows

For this specific model, both constructions resolve to the same 87 unique
analytical surfaces and pass the same representative material-point checks.
The hierarchy compiler emits 82 flattened cell instances because each pin is
represented as a separate placed instance. The explicit Boolean model emits
10 cells because the 37 fuel cylinders are one disconnected union cell and the
37 cladding annuli are another disconnected union cell. This is a difference
in CSG representation, not a change in the physical geometry.

Neither script needs Trimesh or PyVista. The transport-code exporters consume
the analytical CSG representation directly. Mesh dependencies are only needed
for primitive previews, visualization and full-scene STL exports.

## Native lattice element replacements

[`lattice_with_replacements.py`](lattice_with_replacements.py) demonstrates a
17x17 default fuel-pin lattice with guide-tube universes substituted at selected
indices. Index tuples are MiniGeom's natural `(i, j, k)` ordering. The script
compiles the geometry analytically and prints a small summary; use installed
OpenMC/TRIPOLI-5 and their respective exporters for engine-native export.

## Modular volume archives

See `save_volume` / `load_volume` and the `Scene.save_volume()` / `Scene.load_volume()` methods for archiving analytical component definitions as `.mgz` files and reusing them in a final assembly script.
