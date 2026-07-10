"""
generate_dummy_data.py
-----------------------
Generates a fully synthetic player-season dataset that MIMICS the shape of a
real sports-analytics dataset (batting/pitching/salary records) without using
any real data or unpublished results. Safe for public repos.

Run:
    python src/generate_dummy_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

N_PLAYERS = 400
YEARS = list(range(2005, 2024))
TEAMS = [f"TM{i:02d}" for i in range(1, 16)]


def make_player_ids(n):
    return [f"player_{i:04d}" for i in range(n)]


def generate_hitters(n_players=N_PLAYERS):
    rows = []
    players = make_player_ids(n_players)
    for pid in players:
        career_start = RNG.choice(YEARS[:-3])
        n_seasons = RNG.integers(3, 12)
        skill = RNG.normal(0.72, 0.05)          # baseline OPS talent
        steadiness = RNG.uniform(0.03, 0.20)    # personal volatility
        for i in range(n_seasons):
            year = career_start + i
            if year not in YEARS:
                break
            ab = RNG.integers(150, 550)
            ops = max(0.4, RNG.normal(skill, steadiness))
            salary = np.exp(RNG.normal(14 + 0.5 * i, 0.6)) * (1 + ops)
            rows.append({
                "player_id": pid,
                "year": year,
                "team": RNG.choice(TEAMS),
                "at_bats": ab,
                "ops": round(ops, 3),
                "salary": round(salary, 2),
            })
    return pd.DataFrame(rows)


def generate_pitchers(n_players=N_PLAYERS):
    rows = []
    players = make_player_ids(n_players)
    for pid in players:
        career_start = RNG.choice(YEARS[:-3])
        n_seasons = RNG.integers(3, 12)
        skill = RNG.normal(4.10, 0.5)
        steadiness = RNG.uniform(0.10, 0.45)
        for i in range(n_seasons):
            year = career_start + i
            if year not in YEARS:
                break
            ip = RNG.integers(40, 210)
            era = max(1.5, RNG.normal(skill, steadiness))
            salary = np.exp(RNG.normal(14 + 0.4 * i, 0.6)) * (1 + (6 - era) / 6)
            rows.append({
                "player_id": pid,
                "year": year,
                "team": RNG.choice(TEAMS),
                "innings_pitched": ip,
                "era": round(era, 3),
                "salary": round(max(salary, 1), 2),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    hitters = generate_hitters()
    pitchers = generate_pitchers()
    hitters.to_csv(OUT_DIR / "dummy_hitters.csv", index=False)
    pitchers.to_csv(OUT_DIR / "dummy_pitchers.csv", index=False)
    print(f"Wrote {len(hitters)} hitter rows and {len(pitchers)} pitcher rows to {OUT_DIR}")
