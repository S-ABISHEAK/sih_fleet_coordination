"""Conflict Predictor dataset (Section 9.2).

Ground truth comes directly from ``ConflictResolver.dependency_edges``,
via the dedicated ``conflict_events`` table (promoted out of
``algorithm_events``' JSONB — see ``storage/models.py::ConflictEvent``)
rather than a derived proxy: a ``(yielder, winner)`` edge at time t **is**
a real, already-arbitrated conflict between that exact pair, straight
from the real Karma/MD-PIBT implementation. One row per (robot pair,
observed snapshot); label = whether that pair appears in any dependency
edge within the horizon window strictly after the observation.
"""
from __future__ import annotations

import json
import math
import uuid
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from fleetnet_sim.datasets.splits import split_for_run

FEATURE_VERSION = "conflict_v1"
DEFAULT_HORIZON_S = 2.0
DEFAULT_PAIR_RADIUS_M = 8.0


@dataclass
class ConflictDatasetResult:
    dataset_id: str
    output_path: str
    n_rows: int
    n_runs: int


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def _load_conflict_index(session: Session, run_ids: list[str]) -> dict[str, list[tuple[float, frozenset]]]:
    q = text(
        "SELECT run_id, simulation_time, yielder_robot_id, winner_robot_id FROM conflict_events "
        "WHERE run_id IN :run_ids"
    ).bindparams(bindparam("run_ids", expanding=True))
    rows = pd.read_sql(q, session.connection(), params={"run_ids": run_ids})
    index: dict[str, list[tuple[float, frozenset]]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        index[row.run_id].append((row.simulation_time, frozenset({row.yielder_robot_id, row.winner_robot_id})))
    for run_id in index:
        index[run_id].sort(key=lambda t: t[0])
    return index


def _pair_conflicts_within_horizon(index: list[tuple[float, frozenset]], pair: frozenset, t_obs: float, horizon_s: float) -> bool:
    times = [t for t, p in index if p == pair]
    if not times:
        return False
    lo = bisect_right(times, t_obs)
    hi = bisect_right(times, t_obs + horizon_s)
    return hi > lo


def build_conflict_dataset(
    session: Session,
    run_ids: list[str],
    output_dir: str,
    horizon_s: float = DEFAULT_HORIZON_S,
    pair_radius_m: float = DEFAULT_PAIR_RADIUS_M,
) -> ConflictDatasetResult:
    states_q = text(
        "SELECT run_id, tick, simulation_time, robot_id, x, y, yaw, speed, task_id, replan_count "
        "FROM robot_state_ts WHERE run_id IN :run_ids"
    ).bindparams(bindparam("run_ids", expanding=True))
    states = pd.read_sql(states_q, session.connection(), params={"run_ids": run_ids})
    if states.empty:
        raise ValueError("no robot_state_ts rows found for the given run_ids")

    conflict_index = _load_conflict_index(session, run_ids)

    rows = []
    for (run_id, tick), snapshot in states.groupby(["run_id", "tick"]):
        sim_time = float(snapshot["simulation_time"].iloc[0])
        recs = snapshot.to_dict("records")
        run_conflicts = conflict_index.get(run_id, [])
        for ra, rb in combinations(recs, 2):
            d = _dist(ra["x"], ra["y"], rb["x"], rb["y"])
            if d > pair_radius_m:
                continue
            pair = frozenset({ra["robot_id"], rb["robot_id"]})
            vax, vay = ra["speed"] * math.cos(ra["yaw"]), ra["speed"] * math.sin(ra["yaw"])
            vbx, vby = rb["speed"] * math.cos(rb["yaw"]), rb["speed"] * math.sin(rb["yaw"])
            rel_vx, rel_vy = vbx - vax, vby - vay
            closing_speed = -((rb["x"] - ra["x"]) * rel_vx + (rb["y"] - ra["y"]) * rel_vy) / max(d, 1e-6)
            label = _pair_conflicts_within_horizon(run_conflicts, pair, sim_time, horizon_s)
            rows.append(
                {
                    "run_id": run_id,
                    "simulation_time": sim_time,
                    "robot_a_id": ra["robot_id"],
                    "robot_b_id": rb["robot_id"],
                    "distance_m": d,
                    "heading_diff_rad": (ra["yaw"] - rb["yaw"] + math.pi) % (2 * math.pi) - math.pi,
                    "speed_a": ra["speed"],
                    "speed_b": rb["speed"],
                    "closing_speed": closing_speed,
                    "same_task": bool(ra["task_id"] and ra["task_id"] == rb["task_id"]),
                    "replan_count_a": ra["replan_count"],
                    "replan_count_b": rb["replan_count"],
                    "label_conflict_within_horizon": label,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no robot pairs fell within pair_radius_m across the given run_ids")
    out["split"] = out["run_id"].map(split_for_run)

    dataset_id = f"conflict_{uuid.uuid4().hex[:12]}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_id}.parquet"
    out.to_parquet(output_path, index=False)

    feature_cols = [c for c in out.columns if c not in ("label_conflict_within_horizon", "split")]
    schema_path = out_dir / f"{dataset_id}.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "model": "conflict",
                "feature_version": FEATURE_VERSION,
                "feature_columns": feature_cols,
                "label_columns": ["label_conflict_within_horizon"],
                "label_definition": f"pair enters a Karma/MD-PIBT-arbitrated conflict within {horizon_s}s of observation",
                "horizon_s": horizon_s,
                "pair_radius_m": pair_radius_m,
                "source_run_ids": run_ids,
                "row_count": len(out),
                "positive_rate": float(out["label_conflict_within_horizon"].mean()),
                "split_counts": out["split"].value_counts().to_dict(),
            },
            indent=2,
        )
    )

    from fleetnet_sim.storage.models import ModelDatasetRegistry

    session.add(
        ModelDatasetRegistry(
            dataset_id=dataset_id,
            model_name="conflict",
            feature_version=FEATURE_VERSION,
            label_definition={"horizon_s": horizon_s, "pair_radius_m": pair_radius_m},
            horizon_s=horizon_s,
            source_run_ids=run_ids,
            output_path=str(output_path),
            status="ready",
        )
    )
    session.commit()

    return ConflictDatasetResult(dataset_id=dataset_id, output_path=str(output_path), n_rows=len(out), n_runs=len(run_ids))
