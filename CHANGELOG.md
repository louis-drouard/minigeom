# Changelog

## 0.10.1

- Cylinder now uses the required `outer_radius` argument and supports annular regions via `inner_radius`.
- CSG compilation of hollow cylinders includes an outer quadric and an oppositely oriented inner quadric.
- Annular cylinder mesh generation added for optional preview/export mesh workflows.
- Legacy 0.10.0 cylinder archives using the `radius` field remain loadable.
- Examples and tests updated to use the new outer-radius terminology.



## 0.10.0

- Added safe, versioned `.mgz` volume archives with JSON data stored in ZIP containers.
- Archives preserve primitive parameters, materials, placements, priorities, explicit Boolean region cells, native lattices, sparse substitutions, and shared graph references.
- Added `save_volume()` / `load_volume()` functions and `Scene.save_volume()` / `Scene.load_volume()` methods.
- Removed STL import into geometry volumes and per-volume, per-instance, and per-node STL export APIs. Full-scene STL export and scene preview remain available as optional mesh features.
- Removed the `MeshPrimitive` volume class and associated import workflow.
- CANDU examples now skip OpenMC and TRIPOLI-5 exports when their respective Python packages are absent.
- Updated CANDU example tests for the current script-based construction API and fixed the test subprocess import path for the `src/` package layout.


## 0.9.1 / 0.9.0

- Sparse per-element lattice replacements with indexing syntax `lattice[i, j, k] = volume`.
- Scene traversal expands replacement universes at the corresponding lattice coordinates.
- OpenMC and TRIPOLI-5 exporters retain native rectangular lattices and sparse replacements.
- Added a 17x17 lattice guide-tube example and exporter regression tests.


## 0.1.0–0.2.0

- Added backend-independent Boolean region expressions, explicit analytical region cells, overlap diagnostics, and the common analytical CSG compiler.
- Added OpenMC geometry export and TRIPOLI-5 AGORA export, including hierarchy, rectangular lattices and material tags.
- Added optional Trimesh, Manifold3D and PyVista tooling for primitive preview and mesh workflows.
- Added equivalent CANDU 37-pin examples in implicit hierarchy and explicit Boolean forms.
