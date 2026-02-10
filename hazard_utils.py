from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


def load_hazard_ensemble(path: str):
    payload = joblib.load(path)
    return payload


def prepare_features(
    x: Dict,
    t_week: int,
    delta: float,
    feature_columns: List[str],
) -> pd.DataFrame:
    row = {**x, "t_week": t_week, "delta": delta, "t_week_sq": t_week**2}
    df = pd.DataFrame([row])
    df = pd.get_dummies(df)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def ensemble_probabilities(
    ensemble_payload,
    x: Dict,
    t_week: int,
    delta: float,
) -> np.ndarray:
    feature_columns = ensemble_payload["feature_columns"]
    features = prepare_features(x, t_week, delta, feature_columns)
    probs = []
    for calibrator in ensemble_payload["calibrators"]:
        probs.append(calibrator.predict_proba(features)[:, 1][0])
    return np.array(probs)


def conservative_probability(probs: np.ndarray, k: float) -> float:
    mean = probs.mean()
    std = probs.std()
    return float(np.clip(mean - k * std, 0.0, 1.0))
