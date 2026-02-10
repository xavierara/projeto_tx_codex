from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_utils import (
    CleaningConfig,
    add_engineered_features,
    add_time_features,
    clean_data,
    fill_categoricals,
    time_based_split,
)


def build_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
            ("numeric", "passthrough", numeric_cols),
        ]
    )
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    config = yaml.safe_load(Path("config.yaml").read_text())
    paths = config["paths"]
    params = config["params"]
    cleaning_cfg = CleaningConfig(**config["cleaning"])
    features_cfg = config["features"]

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
    train_df, valid_df = time_based_split(df, config["split"]["time_split_date"])

    target = np.log1p(train_df["price"])
    numeric_cols = features_cfg["numerics"] + ["month_created"]

    pipeline = build_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(train_df[categorical_cols + numeric_cols], target)

    valid_pred = pipeline.predict(valid_df[categorical_cols + numeric_cols])
    valid_pred_price = np.expm1(valid_pred)
    mae = mean_absolute_error(valid_df["price"], valid_pred_price)
    rmsle = mean_squared_log_error(valid_df["price"], valid_pred_price, squared=False)

    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, paths["price_model"])

    with open(output_dir / "price_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"mae": mae, "rmsle": rmsle}, f, indent=2)

    p0_all = np.expm1(pipeline.predict(df[categorical_cols + numeric_cols]))
    df["p0"] = p0_all
    df[["p0"]].to_parquet(paths["p0_predictions"], index=False)
    joblib.dump({"categorical_cols": categorical_cols, "numeric_cols": numeric_cols}, paths["preprocessing"])


if __name__ == "__main__":
    main()
