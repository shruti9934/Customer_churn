"""Exploratory data analysis: prints key stats and saves charts to reports/figures.

Run with:  python -m churn.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from churn.config import CATEGORICAL_COLS, DATA_PATH, FIGURES_DIR
from churn.data import clean, download_data


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    download_data()
    raw = pd.read_csv(DATA_PATH)
    blank_total = (raw["TotalCharges"].str.strip() == "").sum()
    df = clean(raw)

    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print(f"Blank TotalCharges fixed: {blank_total}")
    print(f"Churn rate: {df['Churn'].mean():.1%}\n")
    print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2), "\n")

    print("Churn rate by category (top differences):")
    gaps = []
    for col in CATEGORICAL_COLS:
        rates = df.groupby(col)["Churn"].mean()
        gaps.append((col, rates.max() - rates.min(), rates.idxmax(), rates.max()))
    for col, gap, worst, rate in sorted(gaps, key=lambda g: -g[1])[:6]:
        print(f"  {col:18s} highest = {worst!r} ({rate:.1%}), spread {gap:.1%}")

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(4, 3.5))
    df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Class balance"); ax.set_ylabel("Customers"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        df.groupby(col)["Churn"].mean().sort_values().plot.barh(ax=ax, color="#DD8452")
        ax.set_title(f"Churn rate by {col}"); ax.set_xlabel("Churn rate"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "churn_by_category.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=36, multiple="stack", ax=axes[0])
    axes[0].set_title("Tenure (months) by churn")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
    axes[1].set_title("Monthly charges by churn")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=False)
    ax.set_title("Correlation")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "correlation.png", dpi=120); plt.close(fig)
    print(f"\nCharts saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
