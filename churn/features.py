"""Feature engineering and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import ADDON_SERVICES, BINARY_NUMERIC_COLS, CATEGORICAL_COLS, NUMERIC_COLS

ENGINEERED_NUMERIC = ["avg_monthly_spend", "num_addon_services", "charge_increase_ratio"]
ENGINEERED_CATEGORICAL = ["tenure_group"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-driven features on top of the raw columns.

    * ``avg_monthly_spend``   - TotalCharges / tenure (what they actually paid per month)
    * ``charge_increase_ratio`` - MonthlyCharges / avg_monthly_spend (>1 means the bill went up)
    * ``num_addon_services``  - how many optional services the customer uses
    * ``tenure_group``        - tenure bucketed into lifecycle stages
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        tenure = X["tenure"].clip(lower=1)
        X["avg_monthly_spend"] = X["TotalCharges"] / tenure
        X.loc[X["tenure"] == 0, "avg_monthly_spend"] = X["MonthlyCharges"]
        X["charge_increase_ratio"] = X["MonthlyCharges"] / X["avg_monthly_spend"].replace(0, np.nan)
        X["charge_increase_ratio"] = X["charge_increase_ratio"].fillna(1.0)
        X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        X["tenure_group"] = pd.cut(
            X["tenure"], bins=[-1, 12, 24, 48, 60, np.inf],
            labels=["0-1y", "1-2y", "2-4y", "4-5y", "5y+"],
        ).astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categoricals."""
    numeric = NUMERIC_COLS + ENGINEERED_NUMERIC
    categorical = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough", BINARY_NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
