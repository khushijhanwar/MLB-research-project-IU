"""
train_model.py
---------------
Trains a simple regression model to predict next-season salary growth from
the engineered consistency feature, and logs the run with MLflow for
reproducibility (params, metrics, and the fitted model artifact).

Run:
    python src/train_model.py
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_pipeline import build_hitters, build_pitchers
from feature_engineering import add_consistency_feature, standardize

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("salary-growth-consistency-demo")


def prep(kind: str) -> pd.DataFrame:
    if kind == "hitters":
        df = build_hitters()
        df = add_consistency_feature(df, value_col="ops", weight_col="at_bats")
    else:
        df = build_pitchers()
        df = add_consistency_feature(df, value_col="era", weight_col="innings_pitched")
    df = df.dropna(subset=["salary_growth_next", "w_cv_to_t"])
    df = standardize(df, ["w_cv_to_t"])
    return df


def run_experiment(kind: str, model_name: str, model):
    df = prep(kind)
    X = df[["w_cv_to_t_z"]]
    y = df["salary_growth_next"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"{kind}-{model_name}"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("kind", kind)
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{kind} | {model_name}] MAE={mae:.4f}  R2={r2:.4f}")
        return {"kind": kind, "model": model_name, "mae": mae, "r2": r2}


if __name__ == "__main__":
    results = []
    for kind in ["hitters", "pitchers"]:
        results.append(run_experiment(kind, "linear_regression", LinearRegression()))
        results.append(run_experiment(
            kind, "random_forest",
            RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
        ))

    print("\nSummary:")
    print(pd.DataFrame(results))
