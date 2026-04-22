"""
models.py
---------
LightGBM model training for household energy forecasting.

Two types of models:
  - Point (MAE) models   : one per cluster + one global (no-clustering baseline)
  - Quantile models      : q=0.10, 0.50, 0.90 per cluster for prediction intervals

All models are trained on 2023 data only.
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

QUANTILES = [0.10, 0.50, 0.90]

_BASE_PARAMS = dict(
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 63,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
)


def _point_params(random_seed: int) -> dict:
    return {**_BASE_PARAMS, "objective": "regression_l1", "metric": "mae",
            "random_state": random_seed}


def _quantile_params(q: float, random_seed: int) -> dict:
    return {**_BASE_PARAMS, "objective": "quantile", "metric": "quantile",
            "alpha": q, "random_state": random_seed}


# ---------------------------------------------------------------------------
# Point models
# ---------------------------------------------------------------------------

def train_cluster_models(
    train_features_clean,
    feature_cols: list,
    cluster_labels: list,
    random_seed: int = 42,
) -> dict:
    """Train one MAE-optimised LightGBM model per cluster.

    Returns
    -------
    dict: cluster_label (int) -> trained LGBMRegressor
    """
    params  = _point_params(random_seed)
    models  = {}

    for label in cluster_labels:
        mask = train_features_clean["cluster"] == label
        df_c = train_features_clean[mask]
        model = lgb.LGBMRegressor(**params)
        model.fit(df_c[feature_cols].values, df_c["consumption"].values)
        models[label] = model
        print(f"  [point] cluster {label}: {len(df_c):,} train rows")

    return models


def train_global_model(
    train_features_clean,
    feature_cols: list,
    random_seed: int = 42,
) -> lgb.LGBMRegressor:
    """Train a single global MAE-optimised model on all regular households.

    This is the dataset-level baseline required by the assignment.
    It is also used as a fallback for households with unmapped cluster labels.
    """
    params = _point_params(random_seed)
    model  = lgb.LGBMRegressor(**params)
    model.fit(
        train_features_clean[feature_cols].values,
        train_features_clean["consumption"].values,
    )
    print(f"  [point] global model: {len(train_features_clean):,} train rows")
    return model


# ---------------------------------------------------------------------------
# Quantile models
# ---------------------------------------------------------------------------

def train_cluster_quantile_models(
    train_features_clean,
    feature_cols: list,
    cluster_labels: list,
    quantiles: list = None,
    random_seed: int = 42,
) -> dict:
    """Train q=0.10, 0.50, 0.90 quantile models per cluster.

    Returns
    -------
    dict: cluster_label -> {quantile -> trained LGBMRegressor}
    """
    if quantiles is None:
        quantiles = QUANTILES

    q_models = {label: {} for label in cluster_labels}

    for label in cluster_labels:
        mask = train_features_clean["cluster"] == label
        df_c = train_features_clean[mask]
        X    = df_c[feature_cols].values
        y    = df_c["consumption"].values
        print(f"  [quantile] cluster {label}: {len(df_c):,} rows")

        for q in quantiles:
            m = lgb.LGBMRegressor(**_quantile_params(q, random_seed))
            m.fit(X, y)
            q_models[label][q] = m
            print(f"    q={q:.2f} ✓")

    return q_models


def train_global_quantile_models(
    train_features_clean,
    feature_cols: list,
    quantiles: list = None,
    random_seed: int = 42,
) -> dict:
    """Train global quantile fallback models.

    Returns
    -------
    dict: {quantile -> trained LGBMRegressor}
    """
    if quantiles is None:
        quantiles = QUANTILES

    X = train_features_clean[feature_cols].values
    y = train_features_clean["consumption"].values
    models = {}

    for q in quantiles:
        m = lgb.LGBMRegressor(**_quantile_params(q, random_seed))
        m.fit(X, y)
        models[q] = m
        print(f"  [quantile] global q={q:.2f} ✓")

    return models


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_point_models(cluster_models: dict, global_model, models_dir: Path) -> None:
    """Save cluster point models and global model as pickle files."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for label, model in cluster_models.items():
        p = models_dir / f"lgbm_cluster_{label}.pkl"
        with open(p, "wb") as f:
            pickle.dump(model, f)

    with open(models_dir / "lgbm_global.pkl", "wb") as f:
        pickle.dump(global_model, f)

    print(f"  Models saved to {models_dir}")


def save_quantile_models(
    cluster_q_models: dict,
    global_q_models: dict,
    models_dir: Path,
) -> None:
    """Save quantile models as pickle files."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for label, q_dict in cluster_q_models.items():
        for q, model in q_dict.items():
            q_str = str(int(q * 100)).zfill(2)
            with open(models_dir / f"lgbm_cluster{label}_q{q_str}.pkl", "wb") as f:
                pickle.dump(model, f)

    for q, model in global_q_models.items():
        q_str = str(int(q * 100)).zfill(2)
        with open(models_dir / f"lgbm_global_q{q_str}.pkl", "wb") as f:
            pickle.dump(model, f)

    n = sum(len(v) for v in cluster_q_models.values()) + len(global_q_models)
    print(f"  {n} quantile model files saved to {models_dir}")
