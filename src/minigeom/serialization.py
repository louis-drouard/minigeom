"""Safe, versioned archive serialization for analytical VolumeDef graphs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import zipfile

from .geometry import (Box, Cone, Cylinder, Ellipsoid, Plane, Quadric, Sphere,
                      Torus, VolumeDef, Placement, LatticePlacement, RegionCellDef)
from .region_expr import RegionExpression, _ShapeLeaf, _BooleanNode



_FORMAT = "MiniGeomArchive"
_FORMAT_VERSION = 1
_PRIMITIVES = {cls.__name__: cls for cls in
               (Box, Cone, Cylinder, Ellipsoid, Plane, Quadric, Sphere, Torus)}


def _json_value(value):
    """Convert primitive field values to JSON-safe values."""
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported serialized value {type(value).__name__}")



def _primitive_to_data(shape):
    if shape is None:
        return None
    if type(shape).__name__ not in _PRIMITIVES:
        raise TypeError(f"Cannot archive unsupported primitive: {type(shape).__name__}")
    return {"type": type(shape).__name__, **{
        key: _json_value(value) for key, value in vars(shape).items()
        if key != "is_bounded"
    }}


def _primitive_from_data(data):
    if data is None:
        return None
    kind = data.get("type")
    cls = _PRIMITIVES.get(kind)
    if cls is None:
        raise ValueError(f"Unknown or unsupported archived primitive type: {kind!r}")
    values = {k: v for k, v in data.items() if k != "type"}
    if kind == "Cylinder" and "outer_radius" not in values and "radius" in values:
        # Compatibility with archives written by MiniGeom 0.10.0 and earlier.
        values["outer_radius"] = values.pop("radius")
    if kind == "Ellipsoid":
        values["radii"] = tuple(values["radii"])
    if kind == "Quadric":
        values["coefficients"] = tuple(values["coefficients"])
        if values.get("bounds") is not None:
            values["bounds"] = tuple(tuple(v) for v in values["bounds"])
    return cls(**values)


def _expr_to_data(expr):
    def visit(node):
        if isinstance(node, _ShapeLeaf):
            return {"op": "primitive", "shape": _primitive_to_data(node.shape),
                    "matrix": list(node.matrix)}
        if isinstance(node, _BooleanNode):
            return {"op": node.operation, "terms": [visit(term) for term in node.terms]}
        raise TypeError(f"Unsupported region expression node: {type(node).__name__}")
    return visit(expr._node)


def _expr_from_data(data):
    def visit(node):
        op = node.get("op")
        if op == "primitive":
            shape = _primitive_from_data(node["shape"])
            matrix = tuple(float(v) for v in node["matrix"])
            if len(matrix) != 16 or not all(math.isfinite(v) for v in matrix):
                raise ValueError("Archived region-expression transform must contain 16 finite values")
            return _ShapeLeaf(shape, matrix)
        if op not in {"intersection", "union", "difference", "complement"}:
            raise ValueError(f"Unknown archived Boolean operation: {op!r}")
        terms = tuple(visit(term) for term in node.get("terms", []))
        expected = 1 if op == "complement" else 2 if op == "difference" else None
        if expected is not None and len(terms) != expected:
            raise ValueError(f"Boolean operation {op!r} expects {expected} terms")
        if op in {"intersection", "union"} and len(terms) < 2:
            raise ValueError(f"Boolean operation {op!r} requires at least two terms")
        return _BooleanNode(op, terms)
    return RegionExpression(visit(data))


def save_volume(volume: VolumeDef, filename: str | Path) -> Path:
    """Save a complete analytical VolumeDef graph to a standalone ``.mgz`` archive."""
    if not isinstance(volume, VolumeDef):
        raise TypeError("save_volume expects a VolumeDef")
    ids: dict[int, str] = {}
    volumes: list[VolumeDef] = []

    def visit(current):
        if id(current) in ids:
            return ids[id(current)]
        uid = f"v{len(ids):06d}"
        ids[id(current)] = uid
        volumes.append(current)
        for placement in current.children:
            visit(placement.child)
        for lattice in current.lattices:
            visit(lattice.child)
            for replacement in lattice.replacements.values():
                visit(replacement)
        return uid

    root_id = visit(volume)
    definitions = []
    for current in volumes:
        item = {
            "id": ids[id(current)], "name": current.name,
            "shape": _primitive_to_data(current.shape),
            "material": current.material, "self_priority": current.self_priority,
            "children": [], "lattices": [], "region_cells": [],
            "next_order": current._next_order,
        }
        for placement in current.children:
            item["children"].append({
                "child": ids[id(placement.child)], "copy_no": placement.copy_no,
                "name": placement.name, "position": list(placement.position),
                "rotation": list(placement.rotation), "priority": placement.priority,
                "material_override": placement.material_override,
                "can_overlap": placement.can_overlap,
                "overlap_priority": placement.overlap_priority,
                "order": placement._order,
            })
        for lattice in current.lattices:
            item["lattices"].append({
                "child": ids[id(lattice.child)], "name": lattice.name,
                "shape": list(lattice.shape), "pitch": list(lattice.pitch),
                "lower_left": list(lattice.lower_left), "priority": lattice.priority,
                "material_override": lattice.material_override, "order": lattice._order,
                "replacements": [
                    {"index": list(index), "volume": ids[id(replacement)]}
                    for index, replacement in sorted(lattice.replacements.items())
                ],
            })
        for cell in current.region_cells:
            item["region_cells"].append({
                "name": cell.name, "region": _expr_to_data(cell.region),
                "material": cell.material, "priority": cell.priority,
                "order": cell._order,
            })
        definitions.append(item)

    payload = {"format": _FORMAT, "format_version": _FORMAT_VERSION,
               "minigeom_version": "0.10.0", "root_volume": root_id,
               "definitions": definitions}
    path = Path(filename)
    if path.suffix.lower() != ".mgz":
        path = path.with_suffix(".mgz")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("geometry.json", encoded)
    return path


def load_volume(filename: str | Path, *, name: str | None = None) -> VolumeDef:
    """Load and reconstruct a standalone archived VolumeDef graph."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if names != ["geometry.json"]:
                raise ValueError("Invalid MiniGeom archive: unexpected entries")
            info = archive.getinfo("geometry.json")
            if info.file_size > 64 * 1024 * 1024:
                raise ValueError("MiniGeom archive is too large (limit: 64 MiB)")
            raw = archive.read("geometry.json")
        data = json.loads(raw.decode("utf-8"))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid MiniGeom archive: {path}") from exc
    if data.get("format") != _FORMAT:
        raise ValueError("Not a MiniGeom archive")
    if data.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"Unsupported MiniGeom archive version: {data.get('format_version')!r}")
    records = data.get("definitions")
    if not isinstance(records, list) or not records:
        raise ValueError("Archive has no volume definitions")
    by_id = {}
    for record in records:
        uid = record.get("id")
        if not isinstance(uid, str) or not uid or uid in by_id:
            raise ValueError("Invalid or duplicate volume ID in archive")
        by_id[uid] = VolumeDef(record["name"], _primitive_from_data(record.get("shape")),
                                record.get("material"), record.get("self_priority", -100))
    # Build graph references only after all definitions exist, preserving shared nodes.
    for record in records:
        current = by_id[record["id"]]
        for p in record.get("children", []):
            if p["child"] not in by_id:
                raise ValueError("Archive placement refers to an unknown volume ID")
            placement = Placement(
                child=by_id[p["child"]], copy_no=p["copy_no"], name=p["name"],
                position=tuple(p["position"]), rotation=tuple(p["rotation"]),
                priority=p["priority"], material_override=p.get("material_override"),
                can_overlap=bool(p.get("can_overlap", False)),
                overlap_priority=int(p.get("overlap_priority", 0)), _order=p.get("order", 0),
            )
            current.children.append(placement)
        for l in record.get("lattices", []):
            if l["child"] not in by_id:
                raise ValueError("Archive lattice refers to an unknown volume ID")
            lattice = LatticePlacement(
                child=by_id[l["child"]], shape=tuple(l["shape"]), pitch=tuple(l["pitch"]),
                lower_left=tuple(l["lower_left"]), priority=l.get("priority", 0),
                name=l["name"], material_override=l.get("material_override"),
                _order=l.get("order", 0), owner=current,
            )
            for replacement in l.get("replacements", []):
                rid = replacement.get("volume")
                if rid not in by_id:
                    raise ValueError("Archive lattice replacement refers to an unknown volume ID")
                i, j, k = replacement.get("index", [])
                lattice.set_element(i, j, k, by_id[rid])
            current.lattices.append(lattice)
        for cell in record.get("region_cells", []):
            current.region_cells.append(RegionCellDef(
                name=cell["name"], region=_expr_from_data(cell["region"]),
                material=cell.get("material"), priority=cell.get("priority", 0),
                _order=cell.get("order", 0),
            ))
        current._next_order = int(record.get("next_order", 0))
    root_id = data.get("root_volume")
    if root_id not in by_id:
        raise ValueError("Archive root volume ID is missing")
    root = by_id[root_id]
    # Reject cyclic dependency graphs, including malicious archives that bypass
    # VolumeDef's placement-cycle validation during construction.
    visited: set[str] = set()
    active: set[str] = set()
    def check_graph(uid):
        if uid in active:
            raise ValueError("Archive contains a cyclic volume dependency graph")
        if uid in visited:
            return
        active.add(uid)
        current_record = next(record for record in records if record["id"] == uid)
        child_ids = [placement["child"] for placement in current_record.get("children", [])]
        for lattice in current_record.get("lattices", []):
            child_ids.append(lattice["child"])
            child_ids.extend(replacement["volume"] for replacement in lattice.get("replacements", []))
        for child_id in child_ids:
            if child_id not in by_id:
                raise ValueError("Archive dependency graph references an unknown volume ID")
            check_graph(child_id)
        active.remove(uid)
        visited.add(uid)
    check_graph(root_id)
    if name is not None:
        if not isinstance(name, str) or not name.strip() or "/" in name:
            raise ValueError("Root override name must be a non-empty name without '/' ")
        root.name = name.strip()
    return root
