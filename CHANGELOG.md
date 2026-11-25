# Changelog

## 0.1.0

- OpenMC geometry export now assigns deterministic numeric material IDs to all non-void symbolic tags.
- Added `materials_ids.json` with tag-to-ID mapping and source metadata (`external` or `openmc.Material`).
- Removed fake OpenMC Material objects from geometry-only export; external IDs are written into the exported `geometry.xml` after OpenMC serializes the geometry.
- `materials.xml` is written only when real `openmc.Material` objects are used by exported cells and material export is enabled.
- `materials_map` may be partial; unbound symbolic tags retain external IDs.
- Added optional explicit `material_ids` for stable IDs across external geometry/material workflows.
- Added a dedicated TRIPOLI-5 AGORA exporter for analytical CSG, nested universes, rectangular lattices, material-name tags, boundary conditions, and `.agora` serialization.
- Corrected the AGORA affine-transform translation so translation-then-rotation matches MiniGeom's `R @ point + translation` convention.
- Set AGORA rectangular-lattice outer universe to `None` when the finite composite envelope already bounds the lattice.
- Added STL import as `MeshPrimitive` and APIs to export local definitions, placed instances, complete node subtrees, and whole scenes.
- Clarified that imported STL meshes are visualization/interchange geometry and are not converted into analytical CSG surfaces.
- Translated remaining French exception messages and diagnostics to English; source comments and docstrings are in English.
- Added tests for external material IDs, partial material binding, STL round-trips, TRIPOLI-5 hierarchy/lattice mapping, and transformed placements.
- Added mesh-backed STL import/export support.
- Added initial TRIPOLI-5 AGORA geometry exporter.
- Added an initial material-ID catalog for symbolic OpenMC material tags.
- Material tags on `VolumeDef` became optional; the default is `None`.
- Added symbolic material tags and optional `openmc.Material` binding.
- Added geometry-only OpenMC export and a CSG JSON description.
- Added finite cones, ellipsoids, ring tori, planes, general quadrics, rectangular lattices, and sampled overlap diagnostics.
- Corrected plane normalization and bounded-quadric clipping.
- Preserved shared `VolumeDef` identity during deep cloning.
