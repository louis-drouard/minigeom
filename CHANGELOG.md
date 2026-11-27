# Changelog

## 0.2.0

- Added sparse per-element replacements to `LatticePlacement`.
- Added indexing syntax `lattice[i, j, k] = volume`, `get_element`,
  `set_element`, and `clear_element`.
- Scene instance traversal now expands replacement universes at the corresponding
  lattice coordinates.
- OpenMC hierarchy export writes replacement universes into native `RectLattice`
  arrays using OpenMC's reversed-Y array convention.
- TRIPOLI-5 hierarchy export assigns replacement universes directly into native
  AGORA rectangular lattices.
- Added a 17x17 lattice guide-tube example and regression tests for both exporters.

## 0.1.0

- Separated the analytical geometry core from optional triangle-mesh tooling. Trimesh is no longer a mandatory dependency and is imported lazily only for mesh/STL operations.
- Added `region_expr.py`, a backend-independent high-level Boolean expression API supporting intersection (`&`), union (`|`), difference (`-`), complement (`~`), translated/rotated operands, `union_all()` and `intersection_all()`.
- Added `VolumeDef.add_cell()` / `add_region_cell()` for explicit analytical cell partitions in a universe.
- Extended `CSGCompiler` to transform, clip, prioritize and serialize explicit region cells using the same analytical surface registry as implicit hierarchy cells.
- Added raw explicit-cell footprints to overlap diagnostics.
- Updated OpenMC and TRIPOLI-5 exporters to select the flat CSG backend automatically when explicit Boolean cells are present; forced hierarchy mode reports the incompatibility.
- Added `examples/candu_hierarchical.py` and `examples/candu_boolean_regions.py`, two equivalent constructions of the CANDU 37-pin assembly.
- Added extensive English comments explaining the set-theoretic and coordinate-transform meaning of Boolean expressions and CSG compilation.
- Moved Trimesh, Manifold3D and PyVista into optional installation extras.
- Added tests for Boolean operations, explicit-cell priority, transformations, domain clipping, exporter fallback and both CANDU examples.
- Fix the open-STL test fixture so it does not call `trimesh.Trimesh.slice_plane()`, which imports the optional SciPy dependency.
- Keep SciPy optional; MiniGeom's STL import path does not require it.
- OpenMC geometry export assigns deterministic numeric material IDs to non-void symbolic tags.
- Added `materials_ids.json` with tag-to-ID mapping and source metadata.
- Added optional real `openmc.Material` binding and conditional `materials.xml` export.
- Added TRIPOLI-5 AGORA exporter for analytical CSG, nested universes, rectangular lattices and material-name tags.
- Added STL import/export for definitions, instances, nodes and scenes.
- Clarified that imported STL meshes are not analytical CSG surfaces.
