"""Basic tests: run with `pytest -q`."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.data import clean, load_data
from churn.explain import explain_customer, predict_proba
from churn.features import FeatureEngineer
from churn.nn_model import TorchMLPClassifier
from churn.train import make_pipeline


def test_clean_fixes_blank_total_charges():
    df = pd.DataFrame({"tenure": [0, 5], "TotalCharges": [" ", "100.5"], "Churn": ["No", "Yes"]})
    out = clean(df)
    assert out["TotalCharges"].tolist() == [0.0, 100.5]
    assert out["Churn"].tolist() == [0, 1]


def test_feature_engineer_adds_columns():
    X, _ = load_data()
    out = FeatureEngineer().transform(X.head(50))
    for col in ["avg_monthly_spend", "num_addon_services", "charge_increase_ratio", "tenure_group"]:
        assert col in out.columns
    assert out["num_addon_services"].between(0, 6).all()


def test_torch_mlp_learns_simple_pattern():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = TorchMLPClassifier(epochs=40).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.85


def test_pipeline_and_explanations_end_to_end():
    X, y = load_data()
    pipe = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    artifact = {"pipeline": pipe, "features": list(X.columns), "background": X.sample(30, random_state=0),
                "threshold": 0.5}
    proba = predict_proba(artifact, X.head(10))
    assert proba.between(0, 1).all()
    drivers = explain_customer(artifact, X.head(1), n_background=30)
    assert len(drivers) == X.shape[1]
    assert {"feature", "value", "contribution"} <= set(drivers.columns)
