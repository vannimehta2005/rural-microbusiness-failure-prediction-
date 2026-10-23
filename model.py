"""
===========================================================================
  model.py — Model training and prediction logic
  Rural Micro-Business Failure Prediction System  (v2 — Hybrid Dataset)
===========================================================================
  load_model()     — train RF + XGBoost on the HYBRID dataset, cached
  predict_single() — inference for one business record
  get_risk_meta()  — probability → (label, css_class, emoji, color)
===========================================================================
"""

import numpy as np
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import (
    FEATURE_COLS, PALETTE, RF_PARAMS,
    RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD, XGB_PARAMS,
)
# ← uses build_hybrid_dataset instead of generate_dataset
from data import build_hybrid_dataset, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — trains ONCE per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model() -> tuple:
    """
    Build the hybrid dataset, preprocess it, then train Random Forest
    (and XGBoost if installed).  Cached for the lifetime of the session.

    Returns
    -------
    rf_model     : trained RandomForestClassifier
    scaler       : fitted StandardScaler
    df           : hybrid DataFrame (used by visualisations)
    results      : {"rf": {"model": ..., "metrics": ...}, "xgb": ...}
    dataset_type : "hybrid" | "synthetic"
    status_msg   : human-readable dataset summary
    """
    df, dataset_type, status_msg = build_hybrid_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    yp_rf  = rf.predict(X_test)
    ypr_rf = rf.predict_proba(X_test)[:, 1]

    rf_metrics = _build_metrics(y_test, yp_rf, ypr_rf)
    results = {"rf": {"model": rf, "metrics": rf_metrics}}

    # ── XGBoost (optional) ────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        xgb_params = {**XGB_PARAMS, "scale_pos_weight": neg / pos}
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        yp_xgb  = xgb.predict(X_test)
        ypr_xgb = xgb.predict_proba(X_test)[:, 1]
        xgb_metrics = _build_metrics(y_test, yp_xgb, ypr_xgb)
        results["xgb"] = {"model": xgb, "metrics": xgb_metrics}

    return rf, scaler, df, results, dataset_type, status_msg


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model, scaler, inputs: dict) -> float:
    """
    Run the trained model on one business record.

    Parameters
    ----------
    model  : trained estimator (RF or XGBoost)
    scaler : fitted StandardScaler
    inputs : dict with keys matching FEATURE_COLS

    Returns
    -------
    float — failure probability in [0, 1]
    """
    row = np.array([[inputs[col] for col in FEATURE_COLS]])
    return float(model.predict_proba(scaler.transform(row))[0, 1])


def get_risk_meta(prob: float) -> tuple:
    """
    Map a failure probability to human-readable risk metadata.

    Returns
    -------
    (label, css_class, emoji, hex_color)
    """
    if prob < RISK_LOW_THRESHOLD:
        return "LOW RISK",    "risk-low",    "🟢", PALETTE["safe"]
    elif prob < RISK_HIGH_THRESHOLD:
        return "MEDIUM RISK", "risk-medium", "🟡", PALETTE["mid"]
    else:
        return "HIGH RISK",   "risk-high",   "🔴", PALETTE["risk"]


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_metrics(y_test, y_pred, y_proba) -> dict:
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "y_pred":    y_pred,
        "cm":        confusion_matrix(y_test, y_pred),
    }
