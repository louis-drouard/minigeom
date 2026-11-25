"""Deterministic sampling-based diagnostics for analytical overlaps.

The checker detects raw primitive intersections by sampling their common AABB,
then checks that each sampled point is assigned to exactly one resolved CSG cell.
It is a diagnostic, not a mathematical proof: very small intersections can be
missed unless the sample count is increased or a targeted point is supplied.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .csg import CSGCompiler, CSGModel, RawInstanceSpec


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(len(indices), dtype=float)
    f = 1.0 / base
    n = indices.copy()
    while np.any(n):
        values += f * (n % base)
        n //= base
        f /= base
    return values


def _halton(count: int, start: int = 1) -> np.ndarray:
    indices = np.arange(start, start + count, dtype=np.int64)
    return np.column_stack((_radical_inverse(indices, 2),
                            _radical_inverse(indices, 3),
                            _radical_inverse(indices, 5)))


@dataclass
class OverlapPair:
    first: str
    second: str
    relationship: str
    sampled_points: int
    raw_overlap_points: int
    resolved_single_cell_points: int
    unresolved_multi_cell_points: int
    unassigned_points: int
    estimated_overlap_volume: float
    estimated_aabb_fraction: float
    example_point: list[float] | None

    @property
    def detected(self) -> bool:
        return self.raw_overlap_points > 0

    @property
    def resolved(self) -> bool:
        return self.unresolved_multi_cell_points == 0 and self.unassigned_points == 0

    @property
    def status(self) -> str:
        if not self.detected:
            return "not_detected"
        return "resolved_by_priority" if self.resolved else "unresolved_partition"

    def to_dict(self):
        data = asdict(self)
        data.update(detected=self.detected, resolved=self.resolved, status=self.status)
        return data


@dataclass
class OverlapReport:
    scene_name: str
    samples_per_pair: int
    candidate_pairs_checked: int
    pair_results: list[OverlapPair]
    world_samples: int
    world_points_tested: int
    world_unassigned_points: int
    world_multi_cell_points: int
    limitations: str = (
        "Deterministic sampling of shared AABBs; small intersections may be missed. "
        "This report is not a formal proof of non-overlap."
    )

    @property
    def detected_overlaps(self):
        return [item for item in self.pair_results if item.detected]

    @property
    def unresolved_overlaps(self):
        return [item for item in self.pair_results if item.detected and not item.resolved]

    @property
    def has_partition_issues(self):
        return bool(self.unresolved_overlaps or self.world_unassigned_points or self.world_multi_cell_points)

    def summary(self) -> dict:
        return {
            "scene": self.scene_name,
            "pairs_checked": self.candidate_pairs_checked,
            "raw_overlaps_detected": len(self.detected_overlaps),
            "overlaps_resolved_by_CSG": sum(p.resolved for p in self.detected_overlaps),
            "unresolved_overlaps": len(self.unresolved_overlaps),
            "world_partition": {
                "samples_requested": self.world_samples,
                "points_inside_world_tested": self.world_points_tested,
                "unassigned_points": self.world_unassigned_points,
                "multiple_cell_points": self.world_multi_cell_points,
            },
            "has_partition_issues": self.has_partition_issues,
        }

    def to_dict(self):
        return {
            "summary": self.summary(),
            "limitations": self.limitations,
            "pairs": [item.to_dict() for item in self.pair_results],
        }

    def export_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


class OverlapChecker:
    """Check raw intersections and resolved partition consistency.

    ``samples`` controls the Halton points per candidate pair. The same number
    (capped at 20,000) is used for the world partition diagnostic.
    """
    def __init__(self, samples: int = 4000, seed: int = 1729, tolerance: float = 1e-9):
        if not isinstance(samples, int) or samples < 64:
            raise ValueError("samples must be an integer >= 64")
        self.samples = samples
        self.seed = int(seed)
        self.tolerance = float(tolerance)

    @staticmethod
    def _relationship(a: RawInstanceSpec, b: RawInstanceSpec) -> str:
        ap, bp = a.path.split("/"), b.path.split("/")
        if len(ap) < len(bp) and bp[:len(ap)] == ap:
            return "parent_child"
        if len(bp) < len(ap) and ap[:len(bp)] == bp:
            return "parent_child"
        if a.parent_path == b.parent_path:
            return "siblings"
        if a.protruding or b.protruding:
            return "cross_branch_with_protrusion"
        return "cross_branch"

    def _intersection_bounds(self, a, b, world):
        lo = np.maximum(np.maximum(np.asarray(a.bounds[0]), np.asarray(b.bounds[0])),
                        np.asarray(world[0]))
        hi = np.minimum(np.minimum(np.asarray(a.bounds[1]), np.asarray(b.bounds[1])),
                        np.asarray(world[1]))
        if np.any(hi <= lo):
            return None
        return lo, hi

    def _evaluate_pair(self, model: CSGModel, a: RawInstanceSpec, b: RawInstanceSpec,
                       points: np.ndarray, box_volume: float) -> OverlapPair:
        raw_hits = single = multi = missing = 0
        overlap_coords = []
        for point in points:
            if not a.region.contains(point, self.tolerance):
                continue
            if not b.region.contains(point, self.tolerance):
                continue
            root = model.raw_instances[0]
            if not root.region.contains(point, self.tolerance):
                continue
            raw_hits += 1
            if len(overlap_coords) < 3:
                overlap_coords.append(point.tolist())
            cell_hits = len(model.find_cells(point, self.tolerance))
            if cell_hits == 1:
                single += 1
            elif cell_hits == 0:
                missing += 1
            else:
                multi += 1
        frac = raw_hits / max(1, len(points))
        return OverlapPair(
            first=a.path, second=b.path, relationship=self._relationship(a,b),
            sampled_points=len(points), raw_overlap_points=raw_hits,
            resolved_single_cell_points=single,
            unresolved_multi_cell_points=multi, unassigned_points=missing,
            estimated_overlap_volume=frac*box_volume,
            estimated_aabb_fraction=frac,
            example_point=overlap_coords[0] if overlap_coords else None,
        )

    def check(self, scene_or_model) -> OverlapReport:
        model = (scene_or_model if isinstance(scene_or_model, CSGModel)
                 else CSGCompiler().compile(scene_or_model))
        raw = model.raw_instances or []
        if not raw or model.world_bounds is None:
            raise ValueError("The CSG model does not contain the raw bounds and instances required for diagnostics")
        sequence = _halton(self.samples + 1, start=max(1, self.seed % 10000))
        pair_results = []
        candidate_pairs_checked = 0
        for i, first in enumerate(raw):
            for second in raw[i+1:]:
                bounds = self._intersection_bounds(first, second, model.world_bounds)
                if bounds is None:
                    continue
                candidate_pairs_checked += 1
                lo, hi = bounds
                box_volume = float(np.prod(hi-lo))
                points = lo + sequence[:self.samples] * (hi-lo)
                result = self._evaluate_pair(model, first, second, points, box_volume)
                pair_results.append(result)

        world_lo, world_hi = map(np.asarray, model.world_bounds)
        world_points = world_lo + sequence[:min(self.samples, 20000)] * (world_hi-world_lo)
        root_region = raw[0].region
        tested = unassigned = multiple = 0
        for point in world_points:
            if not root_region.contains(point, self.tolerance):
                continue
            tested += 1
            hits = len(model.find_cells(point, self.tolerance))
            if hits == 0:
                unassigned += 1
            elif hits > 1:
                multiple += 1

        return OverlapReport(
            scene_name=model.scene_name, samples_per_pair=self.samples,
            candidate_pairs_checked=candidate_pairs_checked,
            pair_results=pair_results, world_samples=len(world_points),
            world_points_tested=tested, world_unassigned_points=unassigned,
            world_multi_cell_points=multiple,
        )
