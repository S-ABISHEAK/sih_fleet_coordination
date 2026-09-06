"""Inference against a trained baseline model, closing the loop this
project's ML pipeline was built for: `simulate` -> telemetry -> dataset
-> `train-model` -> **predict**. Scores a single "snapshot" (one tick of
a stored run) rather than a rerun-time stream — a genuinely live/
streaming predictor would need the engine to call out mid-tick, which is
future work; this validates "if a trained model had been running at
tick X, what would it have said," using only telemetry that existed at
or before that tick (never anything from later, which would be the same
class of leakage the dataset builders themselves guard against).

Feature construction here deliberately mirrors (rather than imports from)
``datasets/*_builder.py``: those build historical *labeled* datasets over
a whole run's grouped telemetry; this needs one unlabeled snapshot at a
single tick. Sharing code would mean threading an "as-of" cutoff through
every builder's grouped-aggregation logic for one call site — more
churn than the small amount of duplicated feature math below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import joblib
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from fleetnet_sim.storage.models import TrainedModelRegistry


def _feature_columns_by_model() -> dict[str, list[str]]:
    """Deferred import (rather than top-level) so importing this module
    never pulls in sklearn unless a model_path (not model_id) lookup
    actually needs it — model_id lookups get feature_columns straight
    from trained_model_registry instead."""
    from fleetnet_sim.models.conflict_model import FEATURE_COLUMNS as CONFLICT_COLUMNS
    from fleetnet_sim.models.congestion_model import FEATURE_COLUMNS as CONGESTION_COLUMNS
    from fleetnet_sim.models.eta_model import FEATURE_COLUMNS as ETA_COLUMNS

    return {"eta": ETA_COLUMNS, "conflict": CONFLICT_COLUMNS, "congestion": CONGESTION_COLUMNS}


@dataclass
class ModelBundle:
    model: object
    model_id: str | None
    model_name: str  # eta | conflict | congestion
    algorithm: str
    feature_columns: list[str]


def load_model_bundle(session: Session, *, model_id: str | None = None, model_path: str | None = None, model_name: str | None = None) -> ModelBundle:
    """Exactly one of ``model_id`` (looked up in ``trained_model_registry``,
    so a prediction can always be traced back to the model that made it)
    or ``model_path`` + ``model_name`` (the model type isn't recoverable
    from a bare .joblib file) must be given."""
    if model_id:
        row = session.get(TrainedModelRegistry, model_id)
        if row is None:
            raise ValueError(f"no such model_id in trained_model_registry: {model_id!r}")
        model = joblib.load(row.model_path)
        return ModelBundle(model=model, model_id=model_id, model_name=row.model_name, algorithm=row.algorithm, feature_columns=row.feature_columns)
    if model_path:
        if not model_name:
            raise ValueError("model_name (eta|conflict|congestion) is required alongside model_path")
        model = joblib.load(model_path)
        feature_columns = _feature_columns_by_model()[model_name]
        return ModelBundle(model=model, model_id=None, model_name=model_name, algorithm=type(model).__name__, feature_columns=feature_columns)
    raise ValueError("must supply exactly one of model_id or (model_path + model_name)")


def _latest_tick(session: Session, table: str, run_id: str, where: str = "") -> int | None:
    q = text(f"SELECT max(tick) FROM {table} WHERE run_id = :run_id {where}")  # noqa: S608 (table is a fixed literal, never user input)
    return session.execute(q, {"run_id": run_id}).scalar_one()


def _local_density(states: pd.DataFrame, radius_m: float = 5.0) -> pd.Series:
    coords = states[["x", "y"]].to_numpy()
    density = []
    for i in range(len(coords)):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        d = (dx**2 + dy**2) ** 0.5
        density.append(int(((d > 1e-9) & (d <= radius_m)).sum()))
    return pd.Series(density, index=states.index)


def predict_eta_snapshot(session: Session, bundle: ModelBundle, run_id: str, at_tick: int | None = None) -> pd.DataFrame:
    conn = session.connection()
    if at_tick is None:
        at_tick = _latest_tick(session, "robot_state_ts", run_id, "AND task_state IN ('TO_PICKUP', 'TO_DROPOFF')")
        if at_tick is None:
            raise ValueError(f"run {run_id!r} has no en-route robot_state_ts rows to predict from")

    states = pd.read_sql(text("SELECT * FROM robot_state_ts WHERE run_id = :r AND tick = :t"), conn, params={"r": run_id, "t": at_tick})
    if states.empty:
        raise ValueError(f"no robot_state_ts rows for run_id={run_id!r} at tick={at_tick}")
    states["local_density"] = _local_density(states)

    active = states[states["task_state"].isin(["TO_PICKUP", "TO_DROPOFF"]) & states["task_id"].notna()]
    if active.empty:
        raise ValueError(f"no robots en route at tick={at_tick} for run_id={run_id!r}")

    tasks = pd.read_sql(
        text("SELECT task_id, task_type, source_zone_id, destination_zone_id, priority, release_time, planned_distance FROM tasks WHERE run_id = :r"),
        conn, params={"r": run_id},
    )
    df = active.merge(tasks, on="task_id", how="inner")
    if df.empty:
        raise ValueError(f"en-route robots at tick={at_tick} reference task_ids not found in tasks (run not finalized yet?)")

    sim_time = float(df["simulation_time"].iloc[0])
    df["age_since_release_s"] = sim_time - df["release_time"]
    df["remaining_distance_m"] = (df["planned_distance"].fillna(0) - df["distance_travelled"]).clip(lower=0)

    X = df[bundle.feature_columns]
    # GradientBoostingRegressor has no non-negativity constraint and can
    # extrapolate below zero (observed in real predictions); a negative
    # remaining time is never physically meaningful, so clip rather than
    # surface it — this is a light post-processing step, not a change to
    # the model or a way of hiding its (honestly-reported, see README)
    # baseline-level accuracy.
    df["predicted_remaining_time_s"] = bundle.model.predict(X).clip(min=0)
    return df[["robot_id", "task_id", "simulation_time", *bundle.feature_columns, "predicted_remaining_time_s"]]


def predict_conflict_snapshot(session: Session, bundle: ModelBundle, run_id: str, at_tick: int | None = None, pair_radius_m: float = 8.0) -> pd.DataFrame:
    conn = session.connection()
    if at_tick is None:
        at_tick = _latest_tick(session, "robot_state_ts", run_id)
        if at_tick is None:
            raise ValueError(f"run {run_id!r} has no robot_state_ts rows to predict from")

    states = pd.read_sql(text("SELECT * FROM robot_state_ts WHERE run_id = :r AND tick = :t"), conn, params={"r": run_id, "t": at_tick})
    if len(states) < 2:
        raise ValueError(f"fewer than 2 robots present at tick={at_tick} for run_id={run_id!r}; no pairs to score")

    rows = []
    for ra, rb in combinations(states.to_dict("records"), 2):
        d = math.hypot(ra["x"] - rb["x"], ra["y"] - rb["y"])
        if d > pair_radius_m:
            continue
        vax, vay = ra["speed"] * math.cos(ra["yaw"]), ra["speed"] * math.sin(ra["yaw"])
        vbx, vby = rb["speed"] * math.cos(rb["yaw"]), rb["speed"] * math.sin(rb["yaw"])
        rel_vx, rel_vy = vbx - vax, vby - vay
        closing_speed = -((rb["x"] - ra["x"]) * rel_vx + (rb["y"] - ra["y"]) * rel_vy) / max(d, 1e-6)
        rows.append(
            {
                "robot_a_id": ra["robot_id"],
                "robot_b_id": rb["robot_id"],
                "simulation_time": ra["simulation_time"],
                "distance_m": d,
                "heading_diff_rad": (ra["yaw"] - rb["yaw"] + math.pi) % (2 * math.pi) - math.pi,
                "speed_a": ra["speed"],
                "speed_b": rb["speed"],
                "closing_speed": closing_speed,
                "same_task": int(bool(ra["task_id"] and ra["task_id"] == rb["task_id"])),
                "replan_count_a": ra["replan_count"],
                "replan_count_b": rb["replan_count"],
            }
        )
    if not rows:
        raise ValueError(f"no robot pairs within pair_radius_m={pair_radius_m} at tick={at_tick} for run_id={run_id!r}")

    df = pd.DataFrame(rows)
    X = df[bundle.feature_columns]
    df["predicted_conflict_probability"] = bundle.model.predict_proba(X)[:, 1]
    df["predicted_conflict"] = bundle.model.predict(X).astype(bool)
    return df


def predict_congestion_snapshot(session: Session, bundle: ModelBundle, run_id: str, at_tick: int | None = None) -> pd.DataFrame:
    conn = session.connection()
    if at_tick is None:
        at_tick = _latest_tick(session, "edge_state_ts", run_id)
        if at_tick is None:
            raise ValueError(f"run {run_id!r} has no edge_state_ts rows to predict from")

    current = pd.read_sql(text("SELECT * FROM edge_state_ts WHERE run_id = :r AND tick = :t"), conn, params={"r": run_id, "t": at_tick})
    if current.empty:
        raise ValueError(f"no edge_state_ts rows for run_id={run_id!r} at tick={at_tick}")

    prev_tick_row = session.execute(
        text("SELECT max(tick) FROM edge_state_ts WHERE run_id = :r AND tick < :t"), {"r": run_id, "t": at_tick}
    ).scalar_one()
    prev = pd.DataFrame()
    if prev_tick_row is not None:
        prev = pd.read_sql(
            text("SELECT edge_source, edge_target, occupancy_count FROM edge_state_ts WHERE run_id = :r AND tick = :t"),
            conn, params={"r": run_id, "t": prev_tick_row},
        )
    if not prev.empty:
        merged = current.merge(prev, on=["edge_source", "edge_target"], how="left", suffixes=("", "_prev"))
        merged["occupancy_delta"] = merged["occupancy_count"] - merged["occupancy_count_prev"].fillna(merged["occupancy_count"])
        current = merged
    else:
        current["occupancy_delta"] = 0

    X = current[bundle.feature_columns]
    current["predicted_congestion_probability"] = bundle.model.predict_proba(X)[:, 1]
    current["predicted_congestion"] = bundle.model.predict(X).astype(bool)
    return current[["edge_source", "edge_target", "simulation_time", *bundle.feature_columns, "predicted_congestion_probability", "predicted_congestion"]]


def predict_snapshot(session: Session, bundle: ModelBundle, run_id: str, at_tick: int | None = None, **kwargs) -> pd.DataFrame:
    if bundle.model_name == "eta":
        return predict_eta_snapshot(session, bundle, run_id, at_tick)
    if bundle.model_name == "conflict":
        return predict_conflict_snapshot(session, bundle, run_id, at_tick, pair_radius_m=kwargs.get("pair_radius_m", 8.0))
    if bundle.model_name == "congestion":
        return predict_congestion_snapshot(session, bundle, run_id, at_tick)
    raise ValueError(f"unknown model_name {bundle.model_name!r}")
