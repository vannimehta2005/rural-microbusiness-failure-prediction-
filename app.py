"""
===========================================================================
  app.py — Streamlit UI entry point
  Rural Micro-Business Failure Prediction System  (v2 — Hybrid Dataset)
===========================================================================
  Run with:
      streamlit run app.py
===========================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from config import (
    APP_LAYOUT, APP_PAGE_ICON, APP_PAGE_TITLE,
    PALETTE, PROJECT_AUTHORS, PROJECT_DEPT, PROJECT_INST, PROJECT_NAME,
)
from model import (
    XGBOOST_AVAILABLE, get_risk_meta, load_model, predict_single,
)
from utils import colour_bar, compute_risk_scores, generate_recommendations
from visualizations import (
    fig_confusion_matrix, fig_feature_importance,
    fig_model_comparison, fig_risk_distribution,
)

# ─── Page config — must be FIRST Streamlit call ───────────────────────────
st.set_page_config(
    page_title=APP_PAGE_TITLE,
    page_icon=APP_PAGE_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #f4f6f9; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a3c6e 0%,#2c5282 60%,#1e3a5f 100%);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] label { color:#c5d5ee !important; font-size:0.84rem !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background-color:#2d4a7a !important; color:#fff !important;
    border:1px solid #4a7fc1 !important; border-radius:6px;
}

[data-testid="stMetric"] {
    background:#fff; border-radius:12px; padding:14px 18px;
    box-shadow:0 2px 10px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] { font-size:0.80rem !important; color:#777 !important; }
[data-testid="stMetricValue"] { font-size:1.75rem !important; font-weight:800 !important; }

.section-header {
    background:linear-gradient(90deg,#1a3c6e,#2980b9); color:white;
    padding:9px 18px; border-radius:8px; font-size:1rem; font-weight:700;
    margin:20px 0 14px 0; letter-spacing:0.2px;
}

.risk-badge {
    display:inline-block; padding:8px 24px; border-radius:30px;
    font-size:1.05rem; font-weight:800; letter-spacing:0.4px;
}
.risk-low    { background:#d5f5e3; color:#1e8449; border:2px solid #2ecc71; }
.risk-medium { background:#fef9e7; color:#b7770d; border:2px solid #f39c12; }
.risk-high   { background:#fadbd8; color:#c0392b; border:2px solid #e74c3c; }

.bar-wrap { background:#e4e9f0; border-radius:8px; height:18px;
            overflow:hidden; margin:6px 0 3px; }
.bar-fill { height:100%; border-radius:8px; }
.bar-label { font-size:0.78rem; color:#666; }

.chart-card { background:#fff; border-radius:12px; padding:18px;
              box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:8px; }

/* dataset badge */
.ds-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:0.78rem; font-weight:700; margin-left:8px;
}
.ds-hybrid    { background:#d5f5e3; color:#1a7a3e; border:1px solid #2ecc71; }
.ds-synthetic { background:#eaf4fb; color:#1a5276; border:1px solid #3498db; }

.footer {
    text-align:center; color:#aaa; font-size:0.76rem;
    margin-top:40px; padding:14px; border-top:1px solid #dee2e8;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION-STATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    """Initialise session-state keys so they exist before first button click."""
    defaults = {
        "prediction_done": False,
        "fail_prob":       None,
        "scores":          None,
        "recs":            None,
        "risk_label":      None,
        "risk_css":        None,
        "risk_emoji":      None,
        "risk_color":      None,
        "monthly_sales":   None,
        # mirror of last submitted inputs (for ratio display)
        "last_inputs":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_session_state()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a3c6e,#2980b9);
                padding:20px 28px;border-radius:12px;margin-bottom:22px;
                box-shadow:0 4px 16px rgba(26,60,110,0.28);">
        <h1 style="color:#fff;margin:0;font-size:1.6rem;font-weight:800;">
            {APP_PAGE_ICON} {PROJECT_NAME}
        </h1>
        <p style="color:#b8d4f0;margin:5px 0 0;font-size:0.88rem;">
            India-Specific AI/ML Early Warning Framework &nbsp;·&nbsp;
            Random Forest + XGBoost &nbsp;·&nbsp; Kaggle + Synthetic Hybrid Dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model (trains ONCE, then cached) ─────────────────────────────
    with st.spinner("🔄  Initialising model and loading dataset …"):
        rf_model, scaler, df, results, dataset_type, dataset_status = load_model()

    # ── Dataset status banner ─────────────────────────────────────────────
    badge_cls  = "ds-hybrid" if dataset_type == "hybrid" else "ds-synthetic"
    badge_text = "Hybrid Dataset" if dataset_type == "hybrid" else "Synthetic Dataset"
    icon       = "🔀" if dataset_type == "hybrid" else "🔬"

    with st.expander(
        f"{icon}  Dataset Status — click to expand",
        expanded=(dataset_type == "synthetic"),
    ):
        st.markdown(
            f'<span class="ds-badge {badge_cls}">{badge_text}</span>',
            unsafe_allow_html=True,
        )
        # Show each line of the status message
        for line in dataset_status.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("❌"):
                st.error(line, icon="❌")
            elif line.startswith("⚠️"):
                st.warning(line, icon="⚠️")
            else:
                st.info(line, icon="ℹ️")

        # Class distribution
        n_safe = int((df["failure"] == 0).sum())
        n_risk = int((df["failure"] == 1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records",       f"{len(df):,}")
        c2.metric("Safe (0)",            f"{n_safe:,}  ({n_safe/len(df)*100:.1f}%)")
        c3.metric("High Risk (1)",       f"{n_risk:,}  ({n_risk/len(df)*100:.1f}%)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Input Panel
    # ════════════════════════════════════════════════════════════════════════
    st.sidebar.markdown("""
    <div style='text-align:center;padding:8px 0 16px;'>
        <div style='font-size:2.2rem;'>🏪</div>
        <div style='font-size:1rem;font-weight:700;color:#e8edf5;'>Business Profile</div>
        <div style='font-size:0.72rem;color:#8aadd4;margin-top:2px;'>
            Fill in details then click Analyse Risk
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("**💰 Sales & Credit**")
    daily_sales = st.sidebar.number_input(
        "Average Daily Sales (₹)", min_value=50.0, max_value=10000.0,
        value=800.0, step=50.0,
        help="Average revenue earned per working day (~26 days/month)")
    credit_given = st.sidebar.number_input(
        "Outstanding Credit Given (₹)", min_value=0.0, max_value=50000.0,
        value=4000.0, step=100.0,
        help="Total informal credit extended to customers (khata dues)")

    st.sidebar.markdown("**📦 Inventory & Loans**")
    inventory_turnover = st.sidebar.number_input(
        "Inventory Turnover (per month)", min_value=0.1, max_value=15.0,
        value=2.5, step=0.1,
        help="How many times stock is fully replenished per month")
    loan_amount = st.sidebar.number_input(
        "Outstanding Loan Amount (₹)", min_value=0.0, max_value=200000.0,
        value=20000.0, step=500.0,
        help="Total outstanding loan balance (any lender)")

    st.sidebar.markdown("**🗓️ Operations**")
    seasonal_choice = st.sidebar.selectbox(
        "Current Season", options=["Off Season", "Peak Season"],
        index=0,
        help="Is the business currently in its peak demand period?")
    seasonal_trend = 1 if seasonal_choice == "Peak Season" else 0

    num_employees = st.sidebar.number_input(
        "Number of Employees (incl. owner)", min_value=1, max_value=20,
        value=2, step=1,
        help="Total workers including yourself")

    st.sidebar.markdown("**🧾 Expenses & Savings**")
    monthly_expenses = st.sidebar.number_input(
        "Monthly Fixed Expenses (₹)", min_value=500.0, max_value=100000.0,
        value=9000.0, step=250.0,
        help="Rent + electricity + staff wages + misc overheads per month")
    savings_buffer = st.sidebar.number_input(
        "Savings / Emergency Fund (₹)", min_value=0.0, max_value=100000.0,
        value=3000.0, step=100.0,
        help="Liquid cash available for emergencies")

    st.sidebar.markdown("---")

    # ── THE FIXED BUTTON ─────────────────────────────────────────────────
    # Prediction runs ONLY when this button is clicked.
    # Results are stored in st.session_state and persist across re-renders.
    analyse_clicked = st.sidebar.button(
        "🔍  Analyse Risk",
        use_container_width=True,
        type="primary",
        help="Click to run the risk prediction with the values above",
    )

    if analyse_clicked:
        inputs = {
            "daily_sales":        daily_sales,
            "credit_given":       credit_given,
            "inventory_turnover": inventory_turnover,
            "loan_amount":        loan_amount,
            "seasonal_trend":     seasonal_trend,
            "num_employees":      num_employees,
            "monthly_expenses":   monthly_expenses,
            "savings_buffer":     savings_buffer,
        }

        # ── Run all computations and store in session_state ───────────────
        fail_prob = predict_single(rf_model, scaler, inputs)
        scores    = compute_risk_scores(inputs)
        recs      = generate_recommendations(inputs, fail_prob, scores)
        risk_label, risk_css, risk_emoji, risk_color = get_risk_meta(fail_prob)

        st.session_state.prediction_done = True
        st.session_state.fail_prob       = fail_prob
        st.session_state.scores          = scores
        st.session_state.recs            = recs
        st.session_state.risk_label      = risk_label
        st.session_state.risk_css        = risk_css
        st.session_state.risk_emoji      = risk_emoji
        st.session_state.risk_color      = risk_color
        st.session_state.monthly_sales   = daily_sales * 26
        st.session_state.last_inputs     = inputs

    st.sidebar.caption(
        "Results appear after clicking Analyse Risk. "
        "Change values and click again to re-assess."
    )

    # ════════════════════════════════════════════════════════════════════════
    # MAIN PANEL — only shown after at least one click
    # ════════════════════════════════════════════════════════════════════════
    if not st.session_state.prediction_done:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:40px;
                    text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.07);
                    margin-top:10px;">
            <div style="font-size:3rem;margin-bottom:12px;">🏪</div>
            <h3 style="color:#1a3c6e;margin-bottom:8px;">
                Ready to Assess Your Business
            </h3>
            <p style="color:#777;font-size:0.95rem;max-width:480px;
                      margin:0 auto;">
                Fill in your business details in the sidebar on the left,
                then click <strong>🔍 Analyse Risk</strong> to see your
                failure probability, risk score, and personalised
                recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Still show visualisations even before first prediction
        st.markdown("<br>", unsafe_allow_html=True)
        _render_visualisations(df, results)

    else:
        # ── Pull stored values ─────────────────────────────────────────────
        fail_prob     = st.session_state.fail_prob
        scores        = st.session_state.scores
        recs          = st.session_state.recs
        risk_label    = st.session_state.risk_label
        risk_css      = st.session_state.risk_css
        risk_emoji    = st.session_state.risk_emoji
        risk_color    = st.session_state.risk_color
        monthly_sales = st.session_state.monthly_sales
        last_inputs   = st.session_state.last_inputs

        # ── SECTION A — Prediction Cards ──────────────────────────────────
        st.markdown('<div class="section-header">📊  Prediction Results</div>',
                    unsafe_allow_html=True)

        col_prob, col_risk, col_cfrs, col_bhi = st.columns([1.7, 1.3, 1.5, 1.5])

        with col_prob:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:18px 20px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.08);
                        border-left:5px solid {risk_color};">
              <div style="font-size:0.78rem;color:#777;margin-bottom:2px;">
                  Failure Probability (90-day horizon)
              </div>
              <div style="font-size:2.6rem;font-weight:900;color:{risk_color};line-height:1.1;">
                  {fail_prob*100:.1f}%
              </div>
              <div class="bar-wrap" style="margin-top:10px;">
                <div class="bar-fill"
                     style="width:{fail_prob*100:.1f}%;background:{risk_color};">
                </div>
              </div>
              <div style="font-size:0.76rem;color:#888;margin-top:3px;">
                  0% = Certain survival &nbsp;·&nbsp; 100% = Certain failure
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_risk:
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:18px 20px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.08);
                        text-align:center;border-top:4px solid {risk_color};">
              <div style="font-size:0.78rem;color:#777;margin-bottom:10px;">Risk Level</div>
              <div class="risk-badge {risk_css}">{risk_emoji} {risk_label}</div>
              <div style="font-size:0.75rem;color:#aaa;margin-top:10px;">
                  Low &lt;35% &nbsp;·&nbsp; Med 35–65% &nbsp;·&nbsp; High &gt;65%
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_cfrs:
            cfrs       = scores["cash_flow_risk"]
            cfrs_color = (PALETTE["safe"] if cfrs < 35 else
                          PALETTE["mid"]  if cfrs < 65 else PALETTE["risk"])
            cfrs_txt   = "Critical" if cfrs > 70 else "Moderate" if cfrs > 40 else "Healthy"
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:18px 20px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.08);
                        border-left:5px solid {cfrs_color};">
              <div style="font-size:0.78rem;color:#777;margin-bottom:2px;">
                  Cash Flow Risk Score
              </div>
              <div style="font-size:2.2rem;font-weight:900;color:{cfrs_color};">{cfrs}</div>
              <div style="font-size:0.76rem;font-weight:700;color:{cfrs_color};margin-bottom:4px;">
                  {cfrs_txt}
              </div>
              {colour_bar(cfrs, low_is_good=True)}
            </div>
            """, unsafe_allow_html=True)

        with col_bhi:
            bhi       = scores["business_health"]
            bhi_color = (PALETTE["risk"] if bhi < 35 else
                         PALETTE["mid"]  if bhi < 65 else PALETTE["safe"])
            bhi_txt   = "Weak" if bhi < 35 else "Moderate" if bhi < 65 else "Strong"
            st.markdown(f"""
            <div style="background:#fff;border-radius:12px;padding:18px 20px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.08);
                        border-left:5px solid {bhi_color};">
              <div style="font-size:0.78rem;color:#777;margin-bottom:2px;">
                  Business Health Index
              </div>
              <div style="font-size:2.2rem;font-weight:900;color:{bhi_color};">{bhi}</div>
              <div style="font-size:0.76rem;font-weight:700;color:{bhi_color};margin-bottom:4px;">
                  {bhi_txt}
              </div>
              {colour_bar(bhi, low_is_good=False)}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SECTION B — Financial Ratios ───────────────────────────────────
        c2s_pct = min(last_inputs["credit_given"] / (monthly_sales + 1) * 100, 999)
        l2m_x   = last_inputs["loan_amount"]      / (monthly_sales + 1)
        e2s_pct = last_inputs["monthly_expenses"] / (monthly_sales + 1) * 100
        sav_mo  = last_inputs["savings_buffer"]   / (last_inputs["monthly_expenses"] + 1)

        st.markdown('<div class="section-header">📐  Key Financial Ratios</div>',
                    unsafe_allow_html=True)
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Monthly Revenue (est.)", f"₹{monthly_sales:,.0f}")
        r2.metric("Credit-to-Sales",        f"{c2s_pct:.1f}%",
                  delta="⚠ High" if c2s_pct > 50 else "✓ OK",
                  delta_color="inverse" if c2s_pct > 50 else "normal")
        r3.metric("Loan-to-Monthly-Sales",  f"{l2m_x:.1f}×",
                  delta="⚠ High" if l2m_x   > 4  else "✓ OK",
                  delta_color="inverse" if l2m_x   > 4  else "normal")
        r4.metric("Expense Ratio",          f"{e2s_pct:.1f}%",
                  delta="⚠ High" if e2s_pct > 60 else "✓ OK",
                  delta_color="inverse" if e2s_pct > 60 else "normal")
        r5.metric("Savings Coverage",       f"{sav_mo:.1f} mo",
                  delta="⚠ Low"  if sav_mo  < 0.5 else "✓ OK",
                  delta_color="inverse" if sav_mo  < 0.5 else "normal")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SECTION C — Recommendations ────────────────────────────────────
        st.markdown('<div class="section-header">📌  Recommendations & Action Plan</div>',
                    unsafe_allow_html=True)
        for kind, title, message in recs:
            full = f"**{title}** — {message}"
            if kind == "error":
                st.error(full,   icon="🆘")
            elif kind == "warning":
                st.warning(full, icon="⚠️")
            elif kind == "success":
                st.success(full, icon="✅")
            else:
                st.info(full,    icon="ℹ️")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Visualisations ────────────────────────────────────────────────
        _render_visualisations(df, results)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION E — Dataset Explorer
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("🗃️  Explore Training Dataset (first 50 rows)", expanded=False):
        display_df           = df.copy()
        display_df["Source"] = "—"
        display_df["Status"] = display_df["failure"].map(
            {0: "✅ Safe", 1: "🔴 High Risk"}
        )
        st.dataframe(
            display_df.drop(columns="failure").head(50),
            use_container_width=True,
            hide_index=True,
        )
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Records",       len(df))
        d2.metric("Safe Businesses",
                  f"{(df['failure']==0).sum()} "
                  f"({(df['failure']==0).mean()*100:.1f}%)")
        d3.metric("High-Risk Businesses",
                  f"{(df['failure']==1).sum()} "
                  f"({(df['failure']==1).mean()*100:.1f}%)")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION F — About
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("ℹ️  About this System", expanded=False):
        st.markdown(f"""
        **{PROJECT_NAME}** is a lightweight AI/ML prototype for rural India's
        informal micro-business sector.

        | Component | Detail |
        |---|---|
        | **Primary Model** | Random Forest — 200 estimators, depth 8, balanced weights |
        | **Secondary Model** | XGBoost — 200 estimators, lr 0.05, scale_pos_weight |
        | **Dataset** | Hybrid: Kaggle credit-risk + 800 synthetic records |
        | **Kaggle source** | laotse/credit-risk-dataset |
        | **Features** | 8 orally collectable financial & operational indicators |
        | **Target** | Binary: 0 = Viable, 1 = High Risk (90-day horizon) |
        | **Deployment** | CSC kiosks, Jan Dhan BC agents, WhatsApp advisory bots |

        **Scoring Formulas:**
        ```
        CFRS = 100 × (0.35·C2S + 0.30·L2S + 0.25·E2S + 0.10·(1−Sav))
        BHI  = 100 × (0.40·Turnover + 0.30·Savings + 0.20·(1−E2S) + 0.10·Season)
        ```
        Built with Python · scikit-learn · XGBoost · Streamlit · kagglehub
        """)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="footer">
        {APP_PAGE_ICON} {PROJECT_NAME} &nbsp;|&nbsp;
        {PROJECT_AUTHORS} &nbsp;|&nbsp;
        {PROJECT_DEPT} &nbsp;|&nbsp;
        {PROJECT_INST} &nbsp;|&nbsp;
        v2 — Hybrid Dataset Edition
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED VISUALISATIONS RENDERER  (called from both pre- and post-prediction)
# ─────────────────────────────────────────────────────────────────────────────

def _render_visualisations(df, results) -> None:
    st.markdown('<div class="section-header">📈  Model Insights & Visualisations</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌟 Feature Importance",
        "📊 Risk Distribution",
        "🔢 Confusion Matrix",
        "⚖️ Model Comparison",
    ])

    with tab1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        rf_model_obj = results["rf"]["model"]
        st.pyplot(fig_feature_importance(rf_model_obj))
        st.caption(
            "Mean decrease in impurity across all 200 Random Forest trees. "
            "Higher score = stronger predictor of business failure."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(fig_risk_distribution(df))
        st.caption(
            "Density distributions of four key features split by failure label "
            "across the hybrid training dataset."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if XGBOOST_AVAILABLE and "xgb" in results:
            cm_c1, cm_c2 = st.columns(2)
            with cm_c1:
                st.pyplot(fig_confusion_matrix(
                    results["rf"]["metrics"]["cm"], "Random Forest",
                    results["rf"]["metrics"]["accuracy"],
                ))
            with cm_c2:
                st.pyplot(fig_confusion_matrix(
                    results["xgb"]["metrics"]["cm"], "XGBoost",
                    results["xgb"]["metrics"]["accuracy"],
                ))
        else:
            st.pyplot(fig_confusion_matrix(
                results["rf"]["metrics"]["cm"], "Random Forest",
                results["rf"]["metrics"]["accuracy"],
            ))
        st.caption("Confusion matrix on held-out test set (20% of hybrid dataset).")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if XGBOOST_AVAILABLE and "xgb" in results:
            st.pyplot(fig_model_comparison(results))
            st.caption("Side-by-side comparison. Dashed line = 80% reference threshold.")
            st.markdown("**Detailed Metrics Table**")
            metrics_df = pd.DataFrame({
                "Model":     ["Random Forest", "XGBoost"],
                "Accuracy":  [f"{results['rf']['metrics']['accuracy']*100:.2f}%",
                              f"{results['xgb']['metrics']['accuracy']*100:.2f}%"],
                "Precision": [f"{results['rf']['metrics']['precision']*100:.2f}%",
                              f"{results['xgb']['metrics']['precision']*100:.2f}%"],
                "Recall":    [f"{results['rf']['metrics']['recall']*100:.2f}%",
                              f"{results['xgb']['metrics']['recall']*100:.2f}%"],
                "F1-Score":  [f"{results['rf']['metrics']['f1']*100:.2f}%",
                              f"{results['xgb']['metrics']['f1']*100:.2f}%"],
                "ROC-AUC":   [f"{results['rf']['metrics']['roc_auc']*100:.2f}%",
                              f"{results['xgb']['metrics']['roc_auc']*100:.2f}%"],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            rf_m = results["rf"]["metrics"]
            st.info("Install XGBoost for model comparison.", icon="ℹ️")
            a, b, c, d, e = st.columns(5)
            a.metric("Accuracy",  f"{rf_m['accuracy']*100:.1f}%")
            b.metric("Precision", f"{rf_m['precision']*100:.1f}%")
            c.metric("Recall",    f"{rf_m['recall']*100:.1f}%")
            d.metric("F1-Score",  f"{rf_m['f1']*100:.1f}%")
            e.metric("ROC-AUC",   f"{rf_m['roc_auc']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
