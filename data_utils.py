from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    price_min: float
    price_max: float
    year_min: int
    year_max: int
    kilometer_min: int
    kilometer_max: int
    powerps_min: int
    powerps_max: int
    outlier_quantiles: Tuple[float, float]


def clean_data(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned[cleaned["price"] > cfg.price_min]
    cleaned = cleaned[cleaned["price"] <= cfg.price_max]
    cleaned = cleaned[cleaned["yearOfRegistration"] >= cfg.year_min]
    cleaned = cleaned[cleaned["yearOfRegistration"] <= cfg.year_max]
    cleaned = cleaned[cleaned["kilometer"] >= cfg.kilometer_min]
    cleaned = cleaned[cleaned["kilometer"] <= cfg.kilometer_max]
    cleaned = cleaned[cleaned["powerPS"] >= cfg.powerps_min]
    cleaned = cleaned[cleaned["powerPS"] <= cfg.powerps_max]

    lower_q, upper_q = cfg.outlier_quantiles
    q_low = cleaned["price"].quantile(lower_q)
    q_high = cleaned["price"].quantile(upper_q)
    cleaned = cleaned[(cleaned["price"] >= q_low) & (cleaned["price"] <= q_high)]
    return cleaned


def fill_categoricals(df: pd.DataFrame, categoricals: Iterable[str]) -> pd.DataFrame:
    filled = df.copy()
    for col in categoricals:
        filled[col] = filled[col].fillna("unknown").astype("category")
    return filled


def add_time_features(df: pd.DataFrame, t_max: int) -> pd.DataFrame:
    enriched = df.copy()
    enriched["dateCreated"] = pd.to_datetime(enriched["dateCreated"], errors="coerce")
    enriched["lastSeen"] = pd.to_datetime(enriched["lastSeen"], errors="coerce")
    enriched["duration_days"] = (
        enriched["lastSeen"] - enriched["dateCreated"]
    ).dt.days
    enriched["duration_weeks"] = np.maximum(
        1, np.ceil(enriched["duration_days"] / 7.0)
    ).astype(int)
    enriched["duration_weeks_capped"] = np.minimum(enriched["duration_weeks"], t_max)
    enriched["month_created"] = enriched["dateCreated"].dt.month
    return enriched


def time_based_split(
    df: pd.DataFrame, split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(split_date)
    train = df[df["dateCreated"] < cutoff].copy()
    valid = df[df["dateCreated"] >= cutoff].copy()
    return train, valid


def add_engineered_features(
    df: pd.DataFrame, current_year: int, postal_bucket_digits: int, use_postal_bucket: bool
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["age_years"] = current_year - enriched["yearOfRegistration"]
    enriched["log_kilometer"] = np.log1p(enriched["kilometer"])
    enriched["log_powerPS"] = np.log1p(enriched["powerPS"])
    if use_postal_bucket and "postalCode" in enriched.columns:
        enriched["postal_bucket"] = (
            enriched["postalCode"].astype(str).str[:postal_bucket_digits]
        )
    return enriched


def person_week_expansion(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        duration = int(row["duration_weeks_capped"])
        for week in range(1, duration + 1):
            label = 1 if week == duration else 0
            record = {col: row[col] for col in feature_cols}
            record["t_week"] = week
            record["y"] = label
            records.append(record)
    return pd.DataFrame.from_records(records)
