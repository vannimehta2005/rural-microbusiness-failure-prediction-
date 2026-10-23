"""
===========================================================================
  config.py — Global constants and configuration
  Rural Micro-Business Failure Prediction System  (v2 — Hybrid Dataset)
===========================================================================
"""

# ─── Feature & target ────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "daily_sales", "credit_given", "inventory_turnover",
    "loan_amount", "seasonal_trend", "num_employees",
    "monthly_expenses", "savings_buffer",
]
TARGET_COL: str = "failure"

# ─── Dataset defaults ────────────────────────────────────────────────────────
DATASET_N_SAMPLES:    int   = 1200
DATASET_RANDOM_STATE: int   = 42
MISSING_VALUE_RATE:   float = 0.03
MISSING_VALUE_COLS: list[str] = ["credit_given", "savings_buffer", "inventory_turnover"]

# ─── Hybrid dataset (Kaggle + synthetic) ────────────────────────────────────
SYNTHETIC_N_SAMPLES: int = 800      # synthetic rows appended to Kaggle rows
HYBRID_RANDOM_STATE: int = 42

# ─── Kaggle ──────────────────────────────────────────────────────────────────
KAGGLE_DATASET_SLUG:    str       = "laotse/credit-risk-dataset"
KAGGLE_REQUIRED_COLS: list[str]   = ["person_income", "loan_amnt", "loan_status"]

# ─── Model hyper-parameters ──────────────────────────────────────────────────
RF_PARAMS: dict = {
    "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 4,
    "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
}
XGB_PARAMS: dict = {
    "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "use_label_encoder": False, "eval_metric": "logloss",
    "random_state": 42, "verbosity": 0,
}
TRAIN_TEST_SPLIT_SIZE:   float = 0.20
TRAIN_TEST_RANDOM_STATE: int   = 42

# ─── Risk score weights ──────────────────────────────────────────────────────
CFRS_WEIGHTS: dict[str, float] = {
    "credit_to_sales": 0.35, "loan_to_qtr_sales": 0.30,
    "expense_to_sales": 0.25, "savings_coverage": 0.10,
}
BHI_WEIGHTS: dict[str, float] = {
    "turnover_score": 0.40, "savings_score": 0.30,
    "cost_efficiency": 0.20, "season_bonus": 0.10,
}
BHI_MAX_TURNOVER: float = 8.0
BHI_MAX_SAVINGS:  float = 10000.0
BHI_SEASON_BONUS: float = 0.10

# ─── Risk level thresholds ───────────────────────────────────────────────────
RISK_LOW_THRESHOLD:  float = 0.35
RISK_HIGH_THRESHOLD: float = 0.65

# ─── Recommendation rule thresholds ─────────────────────────────────────────
REC_CREDIT_HIGH:      float = 0.50
REC_CREDIT_MID:       float = 0.25
REC_LOAN_HIGH:        float = 4.0
REC_LOAN_MID:         float = 2.0
REC_INVENTORY_LOW:    float = 2.0
REC_INVENTORY_HIGH:   float = 6.0
REC_SAVINGS_CRITICAL: float = 0.5
REC_SAVINGS_LOW:      float = 1.5
REC_EXPENSE_HIGH:     float = 0.60

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE: dict[str, str] = {
    "safe": "#2ecc71", "risk": "#e74c3c", "mid": "#f39c12",
    "primary": "#1a3c6e", "accent": "#3498db", "background": "#f4f6f9",
}

# ─── Streamlit page meta ─────────────────────────────────────────────────────
APP_PAGE_TITLE: str = "Rural Micro-Business Failure Prediction"
APP_PAGE_ICON:  str = "🏪"
APP_LAYOUT:     str = "wide"

# ─── Project metadata ────────────────────────────────────────────────────────
PROJECT_NAME:    str = "Rural Micro-Business Failure Prediction System"
PROJECT_AUTHORS: str = "Anuj Kumar · Saharsh Bhardwaj · Vanni Mehta"
PROJECT_DEPT:    str = "Dept. of Computer Science & Technology"
PROJECT_INST:    str = "Maharaja Agrasen Institute of Technology, New Delhi"
