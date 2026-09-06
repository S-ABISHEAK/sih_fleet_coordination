"""Task ETA Predictor dataset (Section 9.3).

One row per sampled ``robot_state_ts`` observation while a robot is
actively en route to a pickup/dropoff (``task_state`` in
``TO_PICKUP``/``TO_DROPOFF``). Label = the task's actual remaining
completion time from that observation instant — only defined (and only
included) for tasks that eventually completed, per Section 19's "ETA
observations after task completion are excluded" (an observation is, by
construction, always strictly before completion here; nothing after
completion is ever included as an observation row).
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from fleetnet_sim.datasets.splits import split_for_run
from fleetnet_sim.storage.models import ModelDatasetRegistry

FEATURE_VERSION = "eta_v1"


@dataclass
class ETADatasetResult:
    dataset_id: str
    output_path: str
    n_rows: int
    n_runs: int


def _local_density(df_states: pd.DataFrame, radius_m: float = 5.0) -> pd.Series:
    """Count of other robots within radius_m, at the same (run_id, tick)."""
    density = pd.Series(0, index=df_states.index)
    for (run_id, tick), group in df_states.groupby(["run_id", "tick"]):
        coords = group[["x", "y"]].to_numpy()
        n = len(coords)
        for local_i in range(n):
            dx = coords[:, 0] - coords[local_i, 0]
            dy = coords[:, 1] - coords[local_i, 1]
            d = (dx**2 + dy**2) ** 0.5
            density.loc[group.index[local_i]] = int(((d > 1e-9) & (d <= radius_m)).sum())
    return density


def build_eta_dataset(session: Session, run_ids: list[str], output_dir: str) -> ETADatasetResult:
    conn = session.connection()
    states_q = text(
        "SELECT run_id, tick, simulation_time, robot_id, x, y, speed, task_id, task_state, replan_count, distance_travelled "
        "FROM robot_state_ts WHERE run_id IN :run_ids AND task_state IN ('TO_PICKUP', 'TO_DROPOFF') AND task_id IS NOT NULL"
    ).bindparams(bindparam("run_ids", expanding=True))
    states = pd.read_sql(states_q, conn, params={"run_ids": run_ids})
    if states.empty:
        raise ValueError("no in-progress robot_state_ts rows found for the given run_ids")

    tasks_q = text(
        "SELECT run_id, task_id, task_type, source_zone_id, destination_zone_id, priority, release_time, "
        "assignment_time, planned_distance, completion_time, final_status FROM tasks WHERE run_id IN :run_ids"
    ).bindparams(bindparam("run_ids", expanding=True))
    tasks = pd.read_sql(tasks_q, conn, params={"run_ids": run_ids})
    completed = tasks[tasks["final_status"] == "completed"]

    df = states.merge(completed, on=["run_id", "task_id"], how="inner", suffixes=("", "_task"))

    # Label: strictly future relative to the observation (guards leakage —
    # asserted, not just assumed).
    df["label_remaining_time_s"] = df["completion_time"] - df["simulation_time"]
    df = df[df["label_remaining_time_s"] > 0]

    df["age_since_release_s"] = df["simulation_time"] - df["release_time"]
    df["remaining_distance_m"] = (df["planned_distance"].fillna(0) - df["distance_travelled"]).clip(lower=0)
    df["local_density"] = _local_density(df)

    feature_cols = [
        "run_id", "robot_id", "task_id", "simulation_time",
        "task_type", "source_zone_id", "destination_zone_id", "priority", "age_since_release_s",
        "x", "y", "speed",
        "remaining_distance_m", "replan_count", "local_density",
    ]
    label_cols = ["label_remaining_time_s", "completion_time"]
    out = df[feature_cols + label_cols].copy()
    out["split"] = out["run_id"].map(split_for_run)

    assert (out["completion_time"] > out["simulation_time"]).all(), "leakage check failed: label timestamp must be strictly future"

    dataset_id = f"eta_{uuid.uuid4().hex[:12]}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_id}.parquet"
    out.to_parquet(output_path, index=False)

    schema_path = out_dir / f"{dataset_id}.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "model": "eta",
                "feature_version": FEATURE_VERSION,
                "feature_columns": feature_cols,
                "label_columns": label_cols,
                "label_definition": "remaining time (seconds) until task completion, observed while robot is en route",
                "source_run_ids": run_ids,
                "row_count": len(out),
                "split_counts": out["split"].value_counts().to_dict(),
            },
            indent=2,
        )
    )

    session.add(
        ModelDatasetRegistry(
            dataset_id=dataset_id,
            model_name="eta",
            feature_version=FEATURE_VERSION,
            label_definition={"definition": "remaining_time_s = completion_time - simulation_time", "horizon_s": None},
            horizon_s=None,
            source_run_ids=run_ids,
            output_path=str(output_path),
            status="ready",
        )
    )
    session.commit()

    return ETADatasetResult(dataset_id=dataset_id, output_path=str(output_path), n_rows=len(out), n_runs=len(run_ids))
