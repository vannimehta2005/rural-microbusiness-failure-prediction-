"""
===========================================================================
  visualizations.py — Chart and plot functions
  Rural Micro-Business Failure Prediction System
===========================================================================
  Responsibilities
  ─────────────────
  • fig_feature_importance()  — horizontal bar chart of RF importances
  • fig_risk_distribution()   — 2×2 histogram grid by failure class
  • fig_confusion_matrix()    — seaborn heatmap confusion matrix
  • fig_model_comparison()    — grouped bar chart RF vs XGBoost
===========================================================================
  All functions return a matplotlib Figure object.
  Render in Streamlit with: st.pyplot(fig_xxx(...))
===========================================================================
"""

import matplotlib
matplotlib.use("Agg")                     # non-interactive backend

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FEATURE_COLS, PALETTE


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_importance(model) -> plt.Figure:
    """
    Horizontal bar chart of Random Forest feature importances.

    Bars above the median importance are coloured in the 'risk' palette
    colour; bars below are coloured in 'accent'.

    Parameters
    ----------
    model : trained RandomForestClassifier

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    labels      = [FEATURE_COLS[i].replace("_", " ").title() for i in indices]
    values      = importances[indices]
    colors      = [
        PALETTE["risk"] if v > np.median(importances) else PALETTE["accent"]
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    bars = ax.barh(labels, values * 100, color=colors,
                   edgecolor="white", linewidth=0.7, height=0.6)

    for bar, val in zip(bars, values * 100):
        ax.text(
            val + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left",
            fontsize=8.5, color=PALETTE["primary"], fontweight="bold",
        )

    ax.set_xlabel("Importance (%)", fontsize=10, color=PALETTE["primary"])
    ax.set_title(
        "Feature Importance — Random Forest",
        fontsize=11, fontweight="bold", color=PALETTE["primary"], pad=10,
    )
    ax.tick_params(colors=PALETTE["primary"], labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE["risk"],   label="Above-median importance"),
            mpatches.Patch(color=PALETTE["accent"], label="Below-median importance"),
        ],
        fontsize=8, framealpha=0.5, loc="lower right",
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RISK DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def fig_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    2×2 grid of density histograms comparing Safe vs High-Risk businesses
    across four key financial features.

    Parameters
    ----------
    df : DataFrame returned by generate_dataset() (must contain 'failure' col)

    Returns
    -------
    matplotlib.figure.Figure
    """
    features = [
        ("daily_sales",        "Daily Sales (₹)"),
        ("credit_given",       "Credit Given (₹)"),
        ("loan_amount",        "Loan Amount (₹)"),
        ("inventory_turnover", "Inventory Turnover"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["background"])
    fig.suptitle(
        "Feature Distribution by Business Status",
        fontsize=12, fontweight="bold", color=PALETTE["primary"],
    )

    for ax, (col, title) in zip(axes.flat, features):
        ax.set_facecolor(PALETTE["background"])
        safe = df.loc[df["failure"] == 0, col].dropna()
        risk = df.loc[df["failure"] == 1, col].dropna()

        ax.hist(safe, bins=28, alpha=0.65, color=PALETTE["safe"],
                label="Safe (0)", density=True, edgecolor="white")
        ax.hist(risk, bins=28, alpha=0.65, color=PALETTE["risk"],
                label="High Risk (1)", density=True, edgecolor="white")

        ax.set_title(title, fontsize=9.5, fontweight="bold", color=PALETTE["primary"])
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.5)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrix(cm, model_name: str, accuracy: float) -> plt.Figure:
    """
    Styled seaborn heatmap confusion matrix.

    Parameters
    ----------
    cm         : numpy confusion matrix (2×2)
    model_name : string label shown in the chart title
    accuracy   : scalar accuracy value in [0, 1]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Safe", "Pred Risk"],
        yticklabels=["Act. Safe",  "Act. Risk"],
        linewidths=1, linecolor="white",
        cbar_kws={"shrink": 0.72},
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}\nAccuracy: {accuracy * 100:.1f}%",
        fontsize=10, fontweight="bold", color=PALETTE["primary"], pad=8,
    )
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_comparison(results: dict) -> plt.Figure:
    """
    Grouped bar chart comparing Random Forest and XGBoost across
    five evaluation metrics.

    Parameters
    ----------
    results : dict from load_model() — must contain "rf" and "xgb" keys

    Returns
    -------
    matplotlib.figure.Figure
    """
    metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names   = list(results.keys())
    x     = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    bar_colors = [PALETTE["accent"], PALETTE["mid"]]

    for i, (key, color) in enumerate(zip(model_names, bar_colors)):
        vals   = [results[key]["metrics"][m] * 100 for m in metric_keys]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars   = ax.bar(
            x + offset, vals, width,
            label=key.upper(), color=color,
            edgecolor="white", alpha=0.9,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=7.5, color=PALETTE["primary"], fontweight="bold",
            )

    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title(
        "Model Comparison: Random Forest vs XGBoost",
        fontsize=11, fontweight="bold", color=PALETTE["primary"], pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.5)
    ax.axhline(y=80, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    return fig
