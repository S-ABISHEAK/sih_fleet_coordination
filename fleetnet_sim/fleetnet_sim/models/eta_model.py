"""Baseline Task-ETA regressor, trained on ``datasets/eta_builder.py``'s
output. Deliberately a simple, well-understood baseline (gradient-boosted
trees over a handful of hand-picked features) — the point of this pass is
proving the full data->model loop works end to end and is honestly
evaluated on a held-out *set of runs* the model never trained on, not
squeezing out state-of-the-art accuracy.

Feature selection excludes ``run_id``/``robot_id``/``task_id`` (identifiers
that don't generalize to a new run) even though the dataset's schema
sidecar lists them as "feature_columns" — those columns exist there for
traceability/joins, not because they're informative for the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.orm import Session

from fleetnet_sim.models.common import TrainResult, load_split_dataset, save_trained_model

NUMERIC_FEATURES = ["age_since_release_s", "x", "y", "speed", "remaining_distance_m", "replan_count", "local_density", "priority"]
CATEGORICAL_FEATURES = ["task_type", "source_zone_id", "destination_zone_id"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN = "label_remaining_time_s"


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=0))])


def _evaluate(pipeline: Pipeline, df: pd.DataFrame) -> dict:
    X, y = df[FEATURE_COLUMNS], df[LABEL_COLUMN]
    pred = pipeline.predict(X)
    return {
        "n_rows": len(df),
        "mae_s": float(mean_absolute_error(y, pred)),
        "rmse_s": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def train_eta_model(session: Session, dataset_path: str, output_dir: str, dataset_id: str | None = None) -> TrainResult:
    splits = load_split_dataset(dataset_path)
    pipeline = _build_pipeline()
    pipeline.fit(splits["train"][FEATURE_COLUMNS], splits["train"][LABEL_COLUMN])

    metrics = {
        "train": _evaluate(pipeline, splits["train"]),
        "val": _evaluate(pipeline, splits["val"]),
        "test": _evaluate(pipeline, splits["test"]),
    }

    return save_trained_model(
        session,
        model=pipeline,
        model_name="eta",
        algorithm="GradientBoostingRegressor",
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        output_dir=output_dir,
    )
