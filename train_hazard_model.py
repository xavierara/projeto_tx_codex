from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from data_utils import (
    CleaningConfig,
    add_engineered_features,
    add_time_features,
    clean_data,
    fill_categoricals,
    person_week_expansion,
)


def build_hazard_features(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    cols = categorical_cols + numeric_cols + ["t_week", "delta"]
    return df[cols]


def train_single_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    config = yaml.safe_load(Path("config.yaml").read_text())
    paths = config["paths"]
    params = config["params"]
    cleaning_cfg = CleaningConfig(**config["cleaning"])
    features_cfg = config["features"]
    hazard_cfg = config["hazard"]

    df = pd.read_csv(paths["raw_data"])
    df = clean_data(df, cleaning_cfg)
    df = add_time_features(df, params["t_max"])
    df = add_engineered_features(
        df,
        current_year=params["current_year"],
        postal_bucket_digits=features_cfg["postal_bucket_digits"],
        use_postal_bucket=features_cfg["use_postal_bucket"],
    )

    categorical_cols = features_cfg["categoricals"]
    if features_cfg["use_postal_bucket"]:
        categorical_cols = categorical_cols + ["postal_bucket"]

    df = fill_categoricals(df, categorical_cols)

    p0_pred = pd.read_parquet(paths["p0_predictions"])["p0"].values
    df["delta"] = np.log(df["price"] / p0_pred)

    feature_cols = categorical_cols + features_cfg["numerics"] + ["month_created", "delta"]
    expanded = person_week_expansion(df, feature_cols)
    expanded["t_week_sq"] = expanded["t_week"] ** 2

    expanded_features = build_hazard_features(
        expanded,
        categorical_cols=categorical_cols,
        numeric_cols=features_cfg["numerics"] + ["month_created", "t_week_sq"],
    )

    expanded_features = pd.get_dummies(expanded_features, columns=categorical_cols, dummy_na=False)

    x_train, x_valid, y_train, y_valid = train_test_split(
        expanded_features,
        expanded["y"],
        test_size=0.2,
        random_state=42,
        stratify=expanded["y"],
    )

    models = []
    calibrators = []
    for idx in range(hazard_cfg["n_models"]):
        bootstrap = x_train.sample(frac=1.0, replace=True, random_state=42 + idx)
        y_bootstrap = y_train.loc[bootstrap.index]
        model = train_single_model(
            bootstrap,
            y_bootstrap,
            scale_pos_weight=hazard_cfg["scale_pos_weight"],
        )
        calibrator = CalibratedClassifierCV(
            model,
            method=hazard_cfg["calibration"],
            cv="prefit",
        )
        calibrator.fit(x_valid, y_valid)
        models.append(model)
        calibrators.append(calibrator)

    probs = np.column_stack([cal.predict_proba(x_valid)[:, 1] for cal in calibrators])
    mean_prob = probs.mean(axis=1)
    metrics = {
        "logloss": log_loss(y_valid, mean_prob),
        "brier": brier_score_loss(y_valid, mean_prob),
    }

    hazard_dir = Path(paths["hazard_models_dir"])
    hazard_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": models, "calibrators": calibrators, "feature_columns": expanded_features.columns.tolist()},
        hazard_dir / "hazard_ensemble.joblib",
    )

    with open(hazard_dir / "hazard_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
