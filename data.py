"""
===========================================================================
  data.py — Dataset generation, Kaggle integration, and preprocessing
  Rural Micro-Business Failure Prediction System  (v2 — Hybrid Dataset)
===========================================================================
  Public API
  ──────────
  load_kaggle_dataset()  → pd.DataFrame | None   (Kaggle real data)
  generate_dataset()     → pd.DataFrame           (synthetic data)
  build_hybrid_dataset() → (pd.DataFrame, str)    (combined + source label)
  preprocess()           → X_train, X_test, y_train, y_test, scaler
===========================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]


from config import (
    DATASET_N_SAMPLES,
    DATASET_RANDOM_STATE,
    FEATURE_COLS,
    HYBRID_RANDOM_STATE,
    KAGGLE_DATASET_SLUG,
    KAGGLE_REQUIRED_COLS,
    MISSING_VALUE_COLS,
    MISSING_VALUE_RATE,
    SYNTHETIC_N_SAMPLES,
    TARGET_COL,
    TRAIN_TEST_RANDOM_STATE,
    TRAIN_TEST_SPLIT_SIZE,
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KAGGLE DATASET INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_kaggle_dataset() -> tuple[pd.DataFrame | None, str]:
    """
    Download the Kaggle credit-risk dataset and convert it to the project
    feature schema.

    Mapping applied
    ───────────────
    person_income  → daily_sales      (annual income ÷ 365 for daily rate)
    loan_amnt      → loan_amount      (direct)
    loan_status    → failure          (direct: 1 = defaulted = high risk)

    Derived / simulated features (added because Kaggle data lacks them)
    ───────────────────────────────────────────────────────────────────
    credit_given       = loan_amount × 0.30
    monthly_expenses   = (person_income / 12) × 0.40
    savings_buffer     = (person_income / 12) × 0.20
    inventory_turnover = Uniform(1, 6)
    seasonal_trend     = Bernoulli(0.45)
    num_employees      = Uniform integer [1, 5]

    Returns
    -------
    (df, status_message)
    df is None if download or mapping fails.
    """
    try:
        import kagglehub  # lazy import so app still runs without it
    except ImportError:
        return None, (
            "⚠️ kagglehub not installed. "
            "Run `pip install kagglehub` to enable Kaggle integration."
        )

    # ── Step 1: download ─────────────────────────────────────────────────────
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
    except Exception as exc:
        return None, (
            f"❌ Kaggle download failed: {exc}\n"
            "Falling back to fully synthetic dataset."
        )

    # ── Step 2: locate CSV ───────────────────────────────────────────────────
    try:
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            return None, (
                f"❌ No CSV file found in downloaded folder: {path}\n"
                "Falling back to fully synthetic dataset."
            )
        csv_path = os.path.join(path, csv_files[0])
        raw = pd.read_csv(csv_path)
    except Exception as exc:
        return None, (
            f"❌ Failed to read CSV: {exc}\n"
            "Falling back to fully synthetic dataset."
        )

    # ── Step 3: validate required columns ────────────────────────────────────
    missing_cols = [c for c in KAGGLE_REQUIRED_COLS if c not in raw.columns]
    if missing_cols:
        return None, (
            f"❌ Kaggle CSV is missing required columns: {missing_cols}\n"
            f"   Available columns: {list(raw.columns)}\n"
            "Falling back to fully synthetic dataset."
        )

    # ── Step 4: column mapping ────────────────────────────────────────────────
    try:
        rng = np.random.RandomState(HYBRID_RANDOM_STATE)
        n   = len(raw)

        # Core mapped columns
        daily_sales  = (raw["person_income"].fillna(raw["person_income"].median()) / 365
                        ).clip(lower=50).values                   # ₹/day equivalent
        loan_amount  = raw["loan_amnt"].fillna(raw["loan_amnt"].median()).clip(lower=0).values
        failure      = raw["loan_status"].fillna(0).astype(int).values

        # Derived features
        annual_income    = raw["person_income"].fillna(raw["person_income"].median()).values
        monthly_income   = annual_income / 12
        credit_given     = loan_amount * 0.30
        monthly_expenses = (monthly_income * 0.40).clip(min=500)
        savings_buffer   = (monthly_income * 0.20).clip(min=0)

        # Simulated features
        inventory_turnover = rng.uniform(1.0, 6.0, n)
        seasonal_trend     = rng.binomial(1, 0.45, n)
        num_employees      = rng.randint(1, 6, n)

        df = pd.DataFrame({
            "daily_sales":        np.round(daily_sales,        2),
            "credit_given":       np.round(credit_given,       2),
            "inventory_turnover": np.round(inventory_turnover, 2),
            "loan_amount":        np.round(loan_amount,        2),
            "seasonal_trend":     seasonal_trend,
            "num_employees":      num_employees,
            "monthly_expenses":   np.round(monthly_expenses,   2),
            "savings_buffer":     np.round(savings_buffer,     2),
            TARGET_COL:           failure,
        })

        # Drop rows where any feature is NaN after mapping
        df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
        df.reset_index(drop=True, inplace=True)

        status = (
            f"✅ Kaggle dataset loaded: {len(df):,} records from `{csv_files[0]}`"
        )
        return df, status

    except Exception as exc:
        return None, (
            f"❌ Column mapping failed: {exc}\n"
            "Falling back to fully synthetic dataset."
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SYNTHETIC DATASET GENERATION  (unchanged logic)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_dataset(
    n_samples: int = DATASET_N_SAMPLES,
    random_state: int = DATASET_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset representing rural Indian
    micro-businesses over a 6-month observation window.

    Parameters
    ----------
    n_samples    : Number of synthetic records to generate.
    random_state : Seed for reproducibility.

    Returns
    -------
    pd.DataFrame with FEATURE_COLS + TARGET_COL columns.
    ~3 % of values in MISSING_VALUE_COLS are set to NaN to simulate
    informal record-keeping gaps typical of the domain.
    """
    rng = np.random.RandomState(random_state)

    # ── Raw feature sampling ──────────────────────────────────────────────
    daily_sales        = rng.uniform(200, 3000, n_samples)
    credit_given       = np.clip(rng.exponential(3000,  n_samples), 0, 15000)
    inventory_turnover = rng.uniform(0.5, 8.0, n_samples)
    loan_amount        = np.clip(rng.exponential(18000, n_samples), 0, 80000)
    seasonal_trend     = rng.binomial(1, 0.45, n_samples)
    num_employees      = rng.randint(1, 6, n_samples)
    monthly_expenses   = rng.uniform(3000, 25000, n_samples)
    savings_buffer     = np.clip(rng.exponential(5000,  n_samples), 0, 30000)

    # ── Derived risk signals ──────────────────────────────────────────────
    monthly_sales          = daily_sales * 26
    credit_to_sales_ratio  = credit_given     / (monthly_sales  + 1e-9)
    loan_to_sales_ratio    = loan_amount      / (monthly_sales  + 1e-9)
    expense_to_sales_ratio = monthly_expenses / (monthly_sales  + 1e-9)
    savings_cover          = savings_buffer   / (monthly_expenses + 1e-9)

    # ── Stochastic failure label ──────────────────────────────────────────
    p_fail = (
        0.30 * np.clip(credit_to_sales_ratio  / 1.0, 0, 1)
      + 0.25 * np.clip(loan_to_sales_ratio    / 3.0, 0, 1)
      + 0.20 * np.clip(expense_to_sales_ratio / 0.8, 0, 1)
      + 0.15 * (1 - np.clip(inventory_turnover / 4.0, 0, 1))
      + 0.10 * (1 - np.clip(savings_cover      / 2.0, 0, 1))
    )
    p_fail = p_fail * (1 - 0.15 * seasonal_trend)
    p_fail = np.clip(p_fail + rng.normal(0, 0.05, n_samples), 0.02, 0.98)
    labels = (rng.rand(n_samples) < p_fail).astype(int)

    df = pd.DataFrame({
        "daily_sales":        np.round(daily_sales,        2),
        "credit_given":       np.round(credit_given,       2),
        "inventory_turnover": np.round(inventory_turnover, 2),
        "loan_amount":        np.round(loan_amount,        2),
        "seasonal_trend":     seasonal_trend,
        "num_employees":      num_employees,
        "monthly_expenses":   np.round(monthly_expenses,   2),
        "savings_buffer":     np.round(savings_buffer,     2),
        TARGET_COL:           labels,
    })

    # Inject realistic missing values
    for col in MISSING_VALUE_COLS:
        mask = rng.rand(n_samples) < MISSING_VALUE_RATE
        df.loc[mask, col] = np.nan

    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HYBRID DATASET BUILDER
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_hybrid_dataset() -> tuple[pd.DataFrame, str, str]:
    """
    Build the training dataset used by the model.

    Strategy
    ────────
    1. Attempt Kaggle download.
    2. Always generate SYNTHETIC_N_SAMPLES synthetic records.
    3. If Kaggle succeeded → concatenate real + synthetic (hybrid).
       If Kaggle failed   → fall back to full synthetic (1,200 records).

    Returns
    -------
    (df, dataset_type, status_message)
    dataset_type : "hybrid" | "synthetic"
    status_message : human-readable summary for the UI info banner
    """
    kaggle_df, kaggle_status = load_kaggle_dataset()

    synthetic_df = generate_dataset(n_samples=SYNTHETIC_N_SAMPLES,
                                    random_state=HYBRID_RANDOM_STATE)

    if kaggle_df is not None and len(kaggle_df) > 0:
        # ── Hybrid: real Kaggle rows + synthetic rows ──────────────────────
        df = pd.concat([kaggle_df, synthetic_df], ignore_index=True)
        df = df.sample(frac=1, random_state=HYBRID_RANDOM_STATE).reset_index(drop=True)
        dataset_type = "hybrid"
        status = (
            f"{kaggle_status}\n"
            f"🔀 **Hybrid dataset built:** "
            f"{len(kaggle_df):,} Kaggle rows + "
            f"{len(synthetic_df):,} synthetic rows = "
            f"**{len(df):,} total records**"
        )
    else:
        # ── Fallback: synthetic-only ───────────────────────────────────────
        df = generate_dataset(n_samples=DATASET_N_SAMPLES,
                              random_state=DATASET_RANDOM_STATE)
        dataset_type = "synthetic"
        status = (
            f"{kaggle_status}\n"
            f"🔬 **Synthetic-only dataset:** {len(df):,} records"
        )

    return df, dataset_type, status


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREPROCESSING 
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    Prepare raw data for model training.

    Steps
    ─────
    1. Impute missing values with column medians.
    2. Separate features (X) and target (y).
    3. Zero-mean, unit-variance feature scaling.
    4. Stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (scaled)
    scaler                           : fitted StandardScaler
    """
    df = df.copy()
    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TRAIN_TEST_SPLIT_SIZE,
        random_state=TRAIN_TEST_RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, scaler




