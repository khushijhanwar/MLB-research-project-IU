"""
app.py — Streamlit dashboard
------------------------------
Automated statistical reporting layer on top of the pipeline: loads engineered
features, fits a quick regression live, and renders interactive charts +
a plain-language summary — the kind of thing that replaces a manually
rebuilt slide deck every reporting cycle.

Run:
    streamlit run dashboard/app.py

Note: all data is synthetic (see src/generate_dummy_data.py). This dashboard
demonstrates the pipeline architecture only, not any real research findings.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import streamlit as st

from data_pipeline import build_hitters, build_pitchers
from feature_engineering import add_consistency_feature, standardize

st.set_page_config(page_title="Performance Consistency & Pay — Demo", layout="wide")

st.title("⚾ Performance Consistency & Salary Growth — Demo Dashboard")
st.caption(
    "Synthetic data only. This is a portfolio demo of the pipeline architecture "
    "(pandas ETL → feature engineering → modeling → reporting), not a publication "
    "of any real research results."
)

player_type = st.sidebar.radio("Player type", ["Hitters", "Pitchers"])
min_periods = st.sidebar.slider("Min qualifying seasons", 3, 6, 3)

@st.cache_data
def load(kind, min_periods):
    if kind == "Hitters":
        df = build_hitters()
        df = add_consistency_feature(df, value_col="ops", weight_col="at_bats")
        perf_col, perf_label = "ops", "OPS"
    else:
        df = build_pitchers()
        df = add_consistency_feature(df, value_col="era", weight_col="innings_pitched")
        perf_col, perf_label = "era", "ERA"
    df = df.dropna(subset=["salary_growth_next", "w_cv_to_t"])
    df = standardize(df, ["w_cv_to_t"])
    return df, perf_col, perf_label

df, perf_col, perf_label = load(player_type, min_periods)

col1, col2, col3 = st.columns(3)
col1.metric("Player-seasons", f"{len(df):,}")
col2.metric("Unique players", f"{df['player_id'].nunique():,}")
col3.metric("Seasons covered", f"{df['year'].min()}–{df['year'].max()}")

tab1, tab2, tab3 = st.tabs(["Overview", "Consistency Explorer", "Model & Report"])

with tab1:
    st.subheader(f"{perf_label} distribution")
    fig = px.histogram(df, x=perf_col, nbins=40, marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Next-season salary growth distribution")
    fig2 = px.histogram(df, x="salary_growth_next", nbins=40, marginal="box")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Consistency (weighted CV) vs. next-season salary growth")
    fig3 = px.scatter(
        df, x="w_cv_to_t", y="salary_growth_next",
        trendline="ols", opacity=0.5,
        labels={"w_cv_to_t": "Weighted CV (inconsistency)", "salary_growth_next": "Salary growth (log, next season)"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Consistency over a player's career (sample)")
    sample_players = df["player_id"].drop_duplicates().sample(min(5, df["player_id"].nunique()), random_state=1)
    sample_df = df[df["player_id"].isin(sample_players)]
    fig4 = px.line(sample_df.sort_values("year"), x="year", y="w_cv_to_t", color="player_id", markers=True)
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Live regression: does consistency predict salary growth?")
    model = smf.ols("salary_growth_next ~ w_cv_to_t_z", data=df).fit(cov_type="HC3")
    beta = model.params["w_cv_to_t_z"]
    pval = model.pvalues["w_cv_to_t_z"]

    m1, m2 = st.columns(2)
    m1.metric("Coefficient (β)", f"{beta:.4f}")
    m2.metric("p-value", f"{pval:.4g}")

    direction = "more consistent → higher" if beta < 0 else "more consistent → lower"
    significance = "statistically significant (p < .05)" if pval < 0.05 else "not statistically significant"
    st.markdown(
        f"**Auto-generated summary:** In this synthetic sample, the relationship is "
        f"{direction} salary growth, and the result is {significance}."
    )

    with st.expander("Full regression output"):
        st.text(model.summary())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download engineered dataset (CSV)", csv, "engineered_features.csv", "text/csv")

st.sidebar.info(
    "Pipeline: `generate_dummy_data.py` → `data_pipeline.py` → "
    "`feature_engineering.py` → this dashboard. See `src/train_model.py` for "
    "the MLflow-tracked model-training version."
)
