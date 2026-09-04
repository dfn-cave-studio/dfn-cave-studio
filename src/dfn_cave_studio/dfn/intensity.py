"""Borehole P10 and direction-corrected P32 inference for M9."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Callable

import numpy as np
from scipy.stats import chi2

from dfn_cave_studio.dfn.fisher import fisher_sample
from dfn_cave_studio.models.borehole import Borehole, BoreholeCollection
from dfn_cave_studio.models.data_management import DomainInterval
from dfn_cave_studio.models.fracture_set import JointSetConfig
from dfn_cave_studio.models.m9 import ObservabilityState, P10Interval, P32Estimate


def _point_at_depth(borehole: Borehole, depth: float) -> np.ndarray:
    points, mds = borehole.compute_trajectory(step_length=0.5)
    depth = float(np.clip(depth, mds[0], mds[-1]))
    return np.array([np.interp(depth, mds, points[:, axis]) for axis in range(3)], dtype=float)


def _trajectory_segments(borehole: Borehole, start: float, end: float) -> list[tuple[float, float, float, float]]:
    """Return unit direction plus physical length for trajectory pieces."""
    points, mds = borehole.compute_trajectory(step_length=0.5)
    cuts = np.unique(np.concatenate(([start, end], mds[(mds > start) & (mds < end)])))
    segments: list[tuple[float, float, float, float]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        pa, pb = _point_at_depth(borehole, float(a)), _point_at_depth(borehole, float(b))
        vector = pb - pa
        length = float(np.linalg.norm(vector))
        if length > 1e-12:
            direction = vector / length
            segments.append((float(direction[0]), float(direction[1]), float(direction[2]), length))
    return segments


def domain_at_depth(hole_id: str, depth: float, intervals: Iterable[DomainInterval]) -> int | None:
    """Return the borehole structural domain at a measured depth."""
    rows = [item for item in intervals if item.hole_id == hole_id]
    terminal_depth = max((item.to_depth for item in rows), default=None)
    matches = [
        item
        for item in rows
        if item.from_depth <= depth < item.to_depth
        or (terminal_depth is not None and depth == terminal_depth and item.to_depth == terminal_depth)
    ]
    return matches[0].domain_id if matches else None


def _fixed_ranges(
    hole_id: str,
    final_depth: float,
    interval_length: float,
    domains: list[DomainInterval],
) -> list[tuple[float, float, int | None, dict[str, object]]]:
    """Split fixed-length sampling ranges at every structural-domain boundary."""
    fixed_edges = sorted(set([*np.arange(0.0, final_depth, interval_length), final_depth]))
    boundaries = sorted(
        {
            float(depth)
            for item in domains
            if item.hole_id == hole_id
            for depth in (item.from_depth, item.to_depth)
            if 0.0 < depth < final_depth
        }
    )
    ranges: list[tuple[float, float, int | None, dict[str, object]]] = []
    for parent_start, parent_end in zip(fixed_edges[:-1], fixed_edges[1:]):
        cuts = [parent_start, *[value for value in boundaries if parent_start < value < parent_end], parent_end]
        was_split = len(cuts) > 2
        for start, end in zip(cuts[:-1], cuts[1:]):
            ranges.append(
                (
                    float(start),
                    float(end),
                    domain_at_depth(hole_id, (start + end) / 2.0, domains),
                    {
                        "interval_mode": "fixed",
                        "split_by_domain_boundary": was_split,
                        "parent_from_depth": float(parent_start),
                        "parent_to_depth": float(parent_end),
                    },
                )
            )
    return ranges


def build_p10_intervals(
    collection: BoreholeCollection,
    roles: dict[str, str],
    domain_intervals: Iterable[DomainInterval],
    *,
    interval_length: float = 10.0,
    interval_mode: str = "fixed",
    set_ids: Iterable[int] | None = None,
) -> list[P10Interval]:
    """Compute P10 intervals from Formal fracture observations only.

    Adjacent intervals are half-open; only the last interval contains its end.
    One result is produced for every observed set in every sampling interval,
    including explicit TRUE_ZERO results.
    """
    if interval_length <= 0:
        raise ValueError("interval_length must be positive")
    domains = list(domain_intervals)
    observed_set_ids = {
        int(obs.set_id)
        for hole in collection.boreholes
        for obs in hole.fracture_observations
        if obs.set_id is not None
    }
    authoritative_set_ids = {int(value) for value in set_ids} if set_ids is not None else observed_set_ids
    if any(value <= 0 for value in authoritative_set_ids):
        raise ValueError("set_ids must contain positive identifiers")
    ordered_set_ids = sorted(authoritative_set_ids)
    output: list[P10Interval] = []
    for hole in collection.boreholes:
        if interval_mode == "domain":
            ranges = [
                (
                    d.from_depth,
                    d.to_depth,
                    d.domain_id,
                    {"interval_mode": "domain", "split_by_domain_boundary": False},
                )
                for d in sorted(
                    (item for item in domains if item.hole_id == hole.borehole_id),
                    key=lambda item: (item.from_depth, item.to_depth),
                )
            ]
        else:
            ranges = _fixed_ranges(hole.borehole_id, hole.collar.final_depth, interval_length, domains)
        for range_index, (start, end, domain_id, provenance) in enumerate(ranges):
            final = range_index == len(ranges) - 1
            segment_directions = _trajectory_segments(hole, start, end)
            physical_length = sum(item[3] for item in segment_directions)
            center = _point_at_depth(hole, (start + end) / 2.0)
            for set_id in ordered_set_ids:
                selected = [
                    obs
                    for obs in hole.fracture_observations
                    if obs.set_id == set_id
                    and obs.measured_depth >= start
                    and (obs.measured_depth <= end if final else obs.measured_depth < end)
                ]
                count = len(selected)
                full_count = sum(obs.has_full_orientation for obs in selected)
                dip_only_count = count - full_count
                unassigned_dip_only = sum(
                    obs.set_id is None
                    and not obs.has_full_orientation
                    and obs.measured_depth >= start
                    and (obs.measured_depth <= end if final else obs.measured_depth < end)
                    for obs in hole.fracture_observations
                )
                output.append(
                    P10Interval(
                        hole_id=hole.borehole_id,
                        from_depth=start,
                        to_depth=end,
                        domain_id=domain_id,
                        set_id=set_id,
                        observation_count=count,
                        sample_length=physical_length,
                        p10=count / physical_length if physical_length > 0 else None,
                        role=roles.get(hole.borehole_id, "calibration"),
                        data_state="true_zero" if count == 0 and physical_length > 0 else "modeled_value",
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        center_z=float(center[2]),
                        segment_directions=segment_directions,
                        provenance={
                            **provenance,
                            "dip_only_set_specific_counted": dip_only_count,
                            "unassigned_dip_only_not_in_set_specific_p10": unassigned_dip_only,
                        },
                        full_orientation_count=full_count,
                        dip_only_count=dip_only_count,
                        unassigned_dip_only_count=unassigned_dip_only,
                    )
                )
    return output


def expected_orientation_exposure(
    joint_set: JointSetConfig,
    direction: tuple[float, float, float],
    *,
    random_seed: int,
    sample_count: int = 20_000,
) -> float:
    """Estimate E(|n dot u|) reproducibly from the set Fisher distribution."""
    rng = np.random.default_rng(random_seed)
    normals = fisher_sample(
        joint_set.orientation.mean_dip_direction,
        joint_set.orientation.mean_dip,
        joint_set.orientation.kappa,
        rng,
        n_samples=sample_count,
    )
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    return float(np.mean(np.abs(normals @ unit)))


def estimate_p32(
    intervals: Iterable[P10Interval],
    joint_sets: Iterable[JointSetConfig],
    *,
    random_seed: int,
    sample_count: int = 20_000,
    low_observability_threshold: float = 0.05,
    joint_sets_by_domain: dict[tuple[int | None, int], JointSetConfig] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[P32Estimate]:
    """Fit domain/set P32 by Poisson MLE using calibration intervals only."""
    sets = {item.set_id: item for item in joint_sets}
    grouped: dict[tuple[int | None, int], list[P10Interval]] = defaultdict(list)
    for interval in intervals:
        if interval.role == "calibration" and interval.set_id is not None:
            grouped[(interval.domain_id, interval.set_id)].append(interval)
    estimates: list[P32Estimate] = []
    ordered_groups = sorted(grouped.items(), key=lambda item: str(item[0]))
    for group_index, ((domain_id, set_id), rows) in enumerate(ordered_groups):
        if cancelled and cancelled():
            raise InterruptedError("density estimation cancelled")
        raw_length = sum(row.sample_length for row in rows)
        count = sum(row.observation_count for row in rows)
        full_count = sum(row.full_orientation_count for row in rows)
        dip_only_count = sum(row.dip_only_count for row in rows)
        selected_set = (
            joint_sets_by_domain.get((domain_id, set_id))
            if joint_sets_by_domain is not None
            else sets.get(set_id)
        )
        if selected_set is None:
            estimates.append(
                P32Estimate(
                    domain_id=domain_id,
                    set_id=set_id,
                    fracture_count=count,
                    full_orientation_count=full_count,
                    dip_only_count=dip_only_count,
                    raw_sample_length=raw_length,
                    effective_sample_length=0.0,
                    mean_exposure=0.0,
                    p32=None,
                    observability=ObservabilityState.INSUFFICIENT_ORIENTATION_DATA,
                    random_seed=random_seed,
                    calibration_holes=sorted({row.hole_id for row in rows}),
                    orientation_model_source=None,
                    eligibility_status="INSUFFICIENT_ORIENTATION_DATA",
                    provenance={
                        "individual_orientation": "missing" if dip_only_count else "available",
                        "orientation_correction_source": None,
                    },
                )
            )
            continue
        effective = 0.0
        for row_index, row in enumerate(rows):
            for dx, dy, dz, length in row.segment_directions:
                exposure = expected_orientation_exposure(
                    selected_set,
                    (dx, dy, dz),
                    random_seed=random_seed + group_index * 1_000_003 + row_index,
                    sample_count=sample_count,
                )
                effective += length * exposure
        mean_exposure = effective / raw_length if raw_length else 0.0
        observable = raw_length > 0 and mean_exposure >= low_observability_threshold
        p32 = count / effective if observable and effective > 0 else None
        se = math.sqrt(count) / effective if observable and count > 0 else (0.0 if observable else None)
        ci_low = 0.5 * chi2.ppf(0.025, 2 * count) / effective if observable and count > 0 else (0.0 if observable else None)
        ci_high = 0.5 * chi2.ppf(0.975, 2 * (count + 1)) / effective if observable else None
        estimates.append(
            P32Estimate(
                domain_id=domain_id,
                set_id=set_id,
                fracture_count=count,
                raw_sample_length=raw_length,
                effective_sample_length=effective,
                mean_exposure=mean_exposure,
                p32=p32,
                standard_error=se,
                ci95_low=ci_low,
                ci95_high=ci_high,
                observability=ObservabilityState.ADEQUATE if observable else ObservabilityState.LOW_OBSERVABILITY,
                random_seed=random_seed,
                calibration_holes=sorted({row.hole_id for row in rows}),
                full_orientation_count=full_count,
                dip_only_count=dip_only_count,
                orientation_model_source="domain_set_model",
                eligibility_status="adequate" if observable else "LOW_OBSERVABILITY",
                provenance={
                    "individual_orientation": "missing" if dip_only_count else "available",
                    "orientation_correction_source": "domain_set_model",
                },
            )
        )
        if progress:
            progress(group_index + 1, len(ordered_groups))
    return estimates
