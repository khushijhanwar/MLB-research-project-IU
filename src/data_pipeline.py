"""
data_pipeline.py
-----------------
Pandas-based ETL pipeline: loads raw player-season CSVs, aggregates to a
clean player-year grain, and derives salary-growth targets.

This mirrors the "reduced data prep time by 60%" pipeline work — it turns
a handful of manual notebook cells into reusable, testable functions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(kind: str) -> pd.DataFrame:
    """kind: 'hitters' or 'pitchers'"""
    path = DATA_DIR / f"dummy_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/generate_dummy_data.py` first."
        )
    return pd.read_csv(path)


def add_salary_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Adds log salary and next-season salary growth, per player."""
    df = df.sort_values(["player_id", "year"]).copy()
    df["log_salary"] = np.log(df["salary"].clip(lower=1))
    df["prev_log_salary"] = df.groupby("player_id")["log_salary"].shift(1)
    df["salary_growth"] = df["log_salary"] - df["prev_log_salary"]
    df["salary_growth_next"] = df.groupby("player_id")["salary_growth"].shift(-1)
    return df


def qualify(df: pd.DataFrame, playing_time_col: str, min_playing_time: int,
            min_seasons: int) -> pd.DataFrame:
    """Keeps player-seasons meeting a playing-time floor, and drops
    players without enough qualifying seasons for a stable estimate."""
    qual = df[df[playing_time_col] >= min_playing_time].copy()
    counts = qual.groupby("player_id")["year"].nunique().rename("n_seasons")
    qual = qual.merge(counts, on="player_id", how="left")
    return qual[qual["n_seasons"] >= min_seasons].copy()


def build_hitters() -> pd.DataFrame:
    raw = load_raw("hitters")
    df = add_salary_growth(raw)
    df = qualify(df, "at_bats", min_playing_time=200, min_seasons=3)
    return df


def build_pitchers() -> pd.DataFrame:
    raw = load_raw("pitchers")
    df = add_salary_growth(raw)
    df = qualify(df, "innings_pitched", min_playing_time=50, min_seasons=3)
    return df


if __name__ == "__main__":
    h = build_hitters()
    p = build_pitchers()
    print("Hitters:", h.shape)
    print("Pitchers:", p.shape)
