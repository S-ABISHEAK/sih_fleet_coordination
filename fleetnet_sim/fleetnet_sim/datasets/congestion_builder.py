"""Congestion Predictor dataset (Section 9.1).

Built from ``edge_state_ts`` (breadth-phase addition — see
``storage/models.py`` and ``core/engine.py::_sample_occupancy``): one row
per sampled (edge, tick) observation across the given runs. Ground truth
is *derived directly from occupancy_count*, not a proxy model — a
congestion event is exactly "``occupancy_count`` on this edge reaches
``congestion_occupancy_threshold`` or more simultaneous robots" at some
sampled tick, which is the concrete thing a width-aware bottleneck (an
aisle occupied by too many robots to pass) shows up as in this telemetry.
Label = does this edge become congested within ``horizon_s`` strictly
after the observation. Same no-future-leakage/run-level-split discipline
as ``eta_builder``/``conflict_builder``.
"""
from __future__ import annotations

import json
import uuid
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from fleetnet_sim.datasets.splits import split_for_run

FEATURE_VERSION = "congestion_v1"
DEFAULT_HORIZON_S = 5.0
DEFAULT_CONGESTION_OCCUPANCY_THRESHOLD = 2


@dataclass
class CongestionDatasetResult:
    dataset_id: str
    output_path: str
    n_rows: int
    n_runs: int


def build_congestion_dataset(
    session: Session,
    run_ids: list[str],
    output_dir: str,
    horizon_s: float = DEFAULT_HORIZON_S,
    congestion_occupancy_threshold: int = DEFAULT_CONGESTION_OCCUPANCY_THRESHOLD,
) -> CongestionDatasetResult:
    q = text(
        "SELECT run_id, tick, simulation_time, edge_source, edge_target, "
        "occupancy_count, avg_speed, min_speed, max_speed FROM edge_state_ts "
        "WHERE run_id IN :run_ids ORDER BY run_id, edge_source, edge_target, simulation_time"
    ).bindparams(bindparam("run_ids", expanding=True))
    df = pd.read_sql(q, session.connection(), params={"run_ids": run_ids})
    if df.empty:
        raise ValueError("no edge_state_ts rows found for the given run_ids")

    # Per-(run_id, edge) sorted (simulation_time, occupancy_count) series,
    # used to look strictly forward of each observation for the label —
    # mirrors conflict_builder's bisect-over-sorted-times approach.
    series: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        series[(row.run_id, row.edge_source, row.edge_target)].append((row.simulation_time, row.occupancy_count))

    rows = []
    df_sorted = df.sort_values(["run_id", "edge_source", "edge_target", "simulation_time"])
    for group_key, group in df_sorted.groupby(["run_id", "edge_source", "edge_target"]):
        times_occ = series[group_key]
        times = [t for t, _ in times_occ]
        occs = [o for _, o in times_occ]
        prev_occupancy = None
        for rec in group.itertuples(index=False):
            t_obs = rec.simulation_time
            lo = bisect_right(times, t_obs)
            hi = bisect_right(times, t_obs + horizon_s)
            future_occs = occs[lo:hi]
            label = any(o >= congestion_occupancy_threshold for o in future_occs)
            rows.append(
                {
                    "run_id": rec.run_id,
                    "tick": rec.tick,
                    "simulation_time": t_obs,
                    "edge_source": rec.edge_source,
                    "edge_target": rec.edge_target,
                    "occupancy_count": rec.occupancy_count,
                    "avg_speed": rec.avg_speed,
                    "min_speed": rec.min_speed,
                    "max_speed": rec.max_speed,
                    "occupancy_delta": 0 if prev_occupancy is None else rec.occupancy_count - prev_occupancy,
                    "label_congestion_within_horizon": bool(label),
                }
            )
            prev_occupancy = rec.occupancy_count

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no edge_state_ts observations produced dataset rows")
    out["split"] = out["run_id"].map(split_for_run)

    dataset_id = f"congestion_{uuid.uuid4().hex[:12]}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_id}.parquet"
    out.to_parquet(output_path, index=False)

    feature_cols = [c for c in out.columns if c not in ("label_congestion_within_horizon", "split")]
    schema_path = out_dir / f"{dataset_id}.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "model": "congestion",
                "feature_version": FEATURE_VERSION,
                "feature_columns": feature_cols,
                "label_columns": ["label_congestion_within_horizon"],
                "label_definition": (
                    f"this edge's occupancy_count reaches >= {congestion_occupancy_threshold} "
                    f"simultaneous robots within {horizon_s}s of observation"
                ),
                "horizon_s": horizon_s,
                "congestion_occupancy_threshold": congestion_occupancy_threshold,
                "source_run_ids": run_ids,
                "row_count": len(out),
                "positive_rate": float(out["label_congestion_within_horizon"].mean()),
                "split_counts": out["split"].value_counts().to_dict(),
            },
            indent=2,
        )
    )

    from fleetnet_sim.storage.models import ModelDatasetRegistry

    session.add(
        ModelDatasetRegistry(
            dataset_id=dataset_id,
            model_name="congestion",
            feature_version=FEATURE_VERSION,
            label_definition={"horizon_s": horizon_s, "congestion_occupancy_threshold": congestion_occupancy_threshold},
            horizon_s=horizon_s,
            source_run_ids=run_ids,
            output_path=str(output_path),
            status="ready",
        )
    )
    session.commit()

    return CongestionDatasetResult(dataset_id=dataset_id, output_path=str(output_path), n_rows=len(out), n_runs=len(run_ids))
