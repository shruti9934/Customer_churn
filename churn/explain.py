"""Per-customer explanations: which inputs pushed this prediction up or down.

Method (model-agnostic, works for every model in the comparison):
for each raw feature, swap the customer's value for values taken from a
background sample of real customers and measure how much the churn
probability changes on average. A positive number means the customer's
actual value *raises* their churn risk compared with a typical customer.
"""
from __future__ import annotations

import pandas as pd


def predict_proba(artifact: dict, X: pd.DataFrame) -> pd.Series:
    proba = artifact["pipeline"].predict_proba(X[artifact["features"]])[:, 1]
    return pd.Series(proba, index=X.index)


def explain_customer(artifact: dict, customer: pd.DataFrame, n_background: int = 100) -> pd.DataFrame:
    """Return a table of feature contributions for a single customer row."""
    assert len(customer) == 1, "pass exactly one customer"
    features = artifact["features"]
    background = artifact["background"].head(n_background)
    pipeline = artifact["pipeline"]
    base = float(pipeline.predict_proba(customer[features])[:, 1][0])

    rows = []
    for feat in features:
        perturbed = pd.concat([customer[features]] * len(background), ignore_index=True)
        perturbed[feat] = background[feat].to_numpy()
        avg_without = float(pipeline.predict_proba(perturbed)[:, 1].mean())
        rows.append({
            "feature": feat,
            "value": customer[feat].iloc[0],
            "contribution": base - avg_without,
        })
    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.7):
        return "High"
    if probability >= threshold:
        return "Medium"
    return "Low"
