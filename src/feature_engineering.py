"""
feature_engineering.py
-----------------------
Builds a rolling, playing-time-weighted "consistency" feature per player
(weighted coefficient of variation up to season t), then standardizes it
with scikit-learn. This is the reusable feature-engineering workflow
referenced in the resume bullet.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rolling_weighted_cv(df: pd.DataFrame, value_col: str, weight_col: str,
                         id_col: str = "player_id", time_col: str = "year",
                         min_periods: int = 3) -> pd.DataFrame:
    """For each player-season, computes the weighted mean, weighted SD,
    and weighted coefficient of variation using all qualifying seasons
    up to and including that year."""
    df = df[[id_col, time_col, value_col, weight_col]].dropna().copy()
    df = df.sort_values([id_col, time_col])

    rows = []
    for pid, g in df.groupby(id_col):
        vals, wts, yrs = g[value_col].values, g[weight_col].values, g[time_col].values
        for i in range(len(g)):
            v, w, y = vals[: i + 1], wts[: i + 1], yrs[i]
            n = len(v)
            if n < min_periods or w.sum() == 0:
                rows.append([pid, y, n, np.nan, np.nan, np.nan])
                continue
            wmean = (v * w).sum() / w.sum()
            wsd = np.sqrt((w * (v - wmean) ** 2).sum() / w.sum())
            wcv = wsd / wmean if wmean != 0 else np.nan
            rows.append([pid, y, n, wmean, wsd, wcv])

    return pd.DataFrame(
        rows,
        columns=[id_col, time_col, "n_seasons_to_t", "w_mean_to_t", "w_sd_to_t", "w_cv_to_t"],
    )


def add_consistency_feature(df: pd.DataFrame, value_col: str, weight_col: str) -> pd.DataFrame:
    roll = rolling_weighted_cv(df, value_col=value_col, weight_col=weight_col)
    merged = df.merge(roll[["player_id", "year", "w_cv_to_t"]], on=["player_id", "year"], how="left")
    return merged


def standardize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Z-scores the given columns using scikit-learn's StandardScaler and
    appends `<col>_z` columns."""
    df = df.dropna(subset=cols).copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols])
    for i, c in enumerate(cols):
        df[f"{c}_z"] = scaled[:, i]
    return df
