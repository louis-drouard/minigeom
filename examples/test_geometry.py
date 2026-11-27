from minigeom.geometry import Box, Cylinder, Sphere
from minigeom.scene import Scene
from minigeom.openmc_exporter import OpenMCHierarchyExporter

def build_test_geometry() -> Scene:
    sc = Scene("test_features")

    # Définitions de volumes (réutilisables)
    monde = sc.define("monde", Box(500, 500, 500), material="air", self_priority=-100)
    assemblage = sc.define("assemblage", Box(140, 140, 140), material="alu", self_priority=-10)

    # 1. Volumes sans overlap (placés loin l'un de l'autre dans `monde`)
    bloc_gauche = sc.define("bloc_gauche", Box(40, 40, 40), material="acier")
    bloc_droit = sc.define("bloc_droit", Box(40, 40, 40), material="cuivre")

    # 2. Volumes en overlap "normal" dans `assemblage`
    cyl_base = sc.define("cyl_base", Cylinder(radius=30, height=80), material="plomb")
    cyl_insert = sc.define("cyl_insert", Cylinder(radius=18, height=90), material="tungstene")

    # 3. Volume réutilisé plusieurs fois à différents endroits
    cible_sph = sc.define("cible_sph", Sphere(radius=14), material="uo2")

    # 4. Volume saillant (overlap inter-VolumeDef) partant de `assemblage` vers `monde`
    sonde_saillante = sc.define("sonde_saillante", Cylinder(radius=7, height=130), material="bore")

    # --- Assemblage ---
    # Overlap "normal" : `cyl_insert` (priorité 20) découpe `cyl_base` (priorité 10)
    assemblage.place(cyl_base, at=(0, 0, 0), priority=10)
    assemblage.place(cyl_insert, at=(10, 0, 0), priority=20)

    # Réutilisation de `cible_sph` (#1 dans `assemblage`)
    assemblage.place(cible_sph, at=(-25, -25, 0), priority=5)

    # Overlap inter-VolumeDef : `sonde_saillante` est définie/placée dans `assemblage`
    # mais s'étend hors de `assemblage` jusque dans `monde` avec priorité saillante
    assemblage.place_protruding(
        sonde_saillante,
        at=(55, 0, 0),
        rotate=(0, 90, 0),
        overlap_priority=50,
        local_priority=15,
    )

    # --- Monde ---
    # Pas d'overlap entre `bloc_gauche` et `bloc_droit` (ni avec `assemblage`)
    monde.place(bloc_gauche, at=(-170, 0, 0), priority=1)
    monde.place(bloc_droit, at=(+170, 0, 0), priority=1)

    # Placement de `assemblage` dans `monde`
    monde.place(assemblage, at=(0, 0, 0), priority=10)

    # Réutilisation du même `VolumeDef` `cible_sph` (#2 directement dans `monde`)
    monde.place(cible_sph, at=(0, 140, 0), priority=5)

    return sc


if __name__ == "__main__":
    sc = build_test_geometry()

    instances = list(sc.iter_instances())
    cache = {}

    for inst in instances:
        raw = inst.world_mesh()
        resolved = sc._mesh_resolved_instances()
        raw_vol = float(raw.volume) if (raw is not None and not raw and raw.is_volume) else 0.0
        res_vol = (
            float(resolved.volume)
            if (resolved is not None and not resolved and resolved.is_volume)
            else 0.0
        )
        print(
            f"{inst.label:<38} | mat={inst.material:<10} | "
            f"protrude={str(inst.can_overlap):<5} | vol_raw={raw_vol:9.1f} -> vol_res={res_vol:9.1f}"
        )

    sc.export_combined_mesh("test_features_resolved.stl")

    sc.view()

    exporter = OpenMCHierarchyExporter(sc)
    exporter.export_xml("openmc_hierarchy_xml")
    exporter.export_summary_json("openmc_hierarchy_xml/hierarchy_summary.json")