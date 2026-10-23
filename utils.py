"""
===========================================================================
  utils.py — Risk scoring, recommendations, and UI helper functions
  Rural Micro-Business Failure Prediction System
===========================================================================
  Responsibilities
  ─────────────────
  • compute_risk_scores()       — CFRS and BHI calculation
  • generate_recommendations()  — rule-based advisory outputs
  • colour_bar()                — HTML progress-bar string for Streamlit
===========================================================================
"""

from config import (
    BHI_MAX_SAVINGS,
    BHI_MAX_TURNOVER,
    BHI_SEASON_BONUS,
    PALETTE,
    REC_CREDIT_HIGH,
    REC_CREDIT_MID,
    REC_EXPENSE_HIGH,
    REC_INVENTORY_HIGH,
    REC_INVENTORY_LOW,
    REC_LOAN_HIGH,
    REC_LOAN_MID,
    REC_SAVINGS_CRITICAL,
    REC_SAVINGS_LOW,
    RISK_HIGH_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_scores(inputs: dict) -> dict:
    """
    Convert raw business inputs into two interpretable scalar risk indices.

    Cash Flow Risk Score (CFRS)  — 0 to 100, higher = worse cash health.
    Business Health Index  (BHI) — 0 to 100, higher = healthier business.

    Parameters
    ----------
    inputs : dict with keys matching FEATURE_COLS

    Returns
    -------
    {"cash_flow_risk": int, "business_health": int}
    """
    monthly_sales = inputs["daily_sales"] * 26

    # Sub-ratios, all clamped to [0, 1]
    c2s = min(inputs["credit_given"]     / (monthly_sales + 1),              1.0)
    l2s = min(inputs["loan_amount"]      / (monthly_sales * 3 + 1),          1.0)
    e2s = min(inputs["monthly_expenses"] / (monthly_sales + 1),              1.0)
    sav = min(inputs["savings_buffer"]   / (inputs["monthly_expenses"] + 1), 1.0)

    # CFRS = 100 × (0.35·C2S + 0.30·L2S + 0.25·E2S + 0.10·(1−Sav))
    cash_flow_risk = int(round(
        (0.35 * c2s + 0.30 * l2s + 0.25 * e2s + 0.10 * (1 - sav)) * 100
    ))

    # BHI = 100 × (0.40·Turnover + 0.30·Savings + 0.20·(1−E2S) + 0.10·Season)
    turnover_score = min(inputs["inventory_turnover"] / BHI_MAX_TURNOVER, 1.0)
    season_bonus   = BHI_SEASON_BONUS if inputs["seasonal_trend"] == 1 else 0.0
    savings_score  = min(inputs["savings_buffer"] / BHI_MAX_SAVINGS, 1.0)

    business_health = int(round(
        (0.40 * turnover_score
       + 0.30 * savings_score
       + 0.20 * (1 - e2s)
       + 0.10 * season_bonus) * 100
    ))

    return {
        "cash_flow_risk":  max(0, min(100, cash_flow_risk)),
        "business_health": max(0, min(100, business_health)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    inputs: dict,
    fail_prob: float,
    scores: dict,
) -> list[tuple[str, str, str]]:
    """
    Rule-based advisory system. Evaluates seven financial threshold rules
    and returns all triggered recommendations.

    Parameters
    ----------
    inputs    : dict with keys matching FEATURE_COLS
    fail_prob : model failure probability in [0, 1]
    scores    : output of compute_risk_scores()

    Returns
    -------
    List of (kind, title, message) tuples where kind is one of:
    "error" | "warning" | "info" | "success"
    """
    recs          = []
    monthly_sales = inputs["daily_sales"] * 26

    c2s            = inputs["credit_given"]     / (monthly_sales + 1)
    l2m            = inputs["loan_amount"]      / (monthly_sales + 1)
    e2s            = inputs["monthly_expenses"] / (monthly_sales + 1)
    months_covered = inputs["savings_buffer"]   / (inputs["monthly_expenses"] + 1)

    # ── R1 / R2 : Credit management ──────────────────────────────────────
    if c2s > REC_CREDIT_HIGH:
        recs.append(("warning", "⚠️ CREDIT RISK",
            "Over 50% of monthly sales locked in unpaid credit. "
            "Collect dues weekly and cap new credit at ₹500 per customer."))
    elif c2s > REC_CREDIT_MID:
        recs.append(("info", "📋 CREDIT CONTROL",
            "Moderate credit extended. Keep a handwritten khata with due dates; "
            "follow up every 15 days."))

    # ── R3 / R4 : Loan management ────────────────────────────────────────
    if l2m > REC_LOAN_HIGH:
        recs.append(("warning", "🏦 LOAN OVEREXPOSURE",
            "Outstanding loan exceeds 4× monthly revenue. "
            "Avoid new borrowing. Consider microfinance restructuring."))
    elif l2m > REC_LOAN_MID:
        recs.append(("info", "💰 LOAN MONITORING",
            "Loan-to-sales ratio is elevated. "
            "Prioritise EMI payments before re-investing in stock."))

    # ── R5 / R6 : Inventory turnover ─────────────────────────────────────
    if inputs["inventory_turnover"] < REC_INVENTORY_LOW:
        recs.append(("warning", "📦 INVENTORY STAGNATION",
            "Stock is moving slowly (< 2 replenishments/month). "
            "Identify slow-moving items; run weekend discount offers to clear stock."))
    elif inputs["inventory_turnover"] > REC_INVENTORY_HIGH:
        recs.append(("success", "✅ INVENTORY EXCELLENT",
            "Excellent stock rotation! Maintain bulk-buying discounts "
            "with your supplier to protect margins."))

    # ── R7 / R8 : Savings buffer ─────────────────────────────────────────
    if months_covered < REC_SAVINGS_CRITICAL:
        recs.append(("warning", "🚨 SAVINGS CRITICAL",
            "Less than 2 weeks of expenses in reserve. "
            "Deposit ₹50–100/day into a savings account as an emergency fund."))
    elif months_covered < REC_SAVINGS_LOW:
        recs.append(("info", "💵 SAVINGS LOW",
            "Build at least 1.5 months of expense cover. "
            "Target saving 10% of daily sales before personal withdrawal."))

    # ── R9 : Seasonal planning ───────────────────────────────────────────
    if inputs["seasonal_trend"] == 0:
        recs.append(("info", "🌧️ OFF-SEASON STRATEGY",
            "Revenue typically dips now. Reduce perishable stock orders by 20–30%; "
            "offer mobile recharge / bill payment as an additional income stream."))

    # ── R10 : Expense ratio ──────────────────────────────────────────────
    if e2s > REC_EXPENSE_HIGH:
        recs.append(("warning", "📉 HIGH EXPENSE RATIO",
            "Costs are eating >60% of revenue. "
            "Review rent, utilities, and staff expenses for renegotiation."))

    # ── R11 : Emergency escalation ───────────────────────────────────────
    if fail_prob >= RISK_HIGH_THRESHOLD:
        recs.append(("error", "🆘 IMMEDIATE ACTION NEEDED",
            "Business is at HIGH RISK of failure within 6 months. "
            "Contact a NABARD-linked microfinance institution or local Krishi Vigyan Kendra "
            "for a free business counselling session."))

    # ── Healthy state (no rules triggered) ───────────────────────────────
    if not recs:
        recs.append(("success", "🎉 BUSINESS IS HEALTHY",
            "No major red flags detected. Keep maintaining your khata records "
            "and build your emergency savings fund progressively."))

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def colour_bar(value: int, low_is_good: bool = True) -> str:
    """
    Return an HTML string rendering a coloured progress bar.

    The bar colour transitions green → amber → red as the value rises
    when low_is_good=True (e.g. CFRS), and the reverse when False (BHI).

    Parameters
    ----------
    value       : integer in [0, 100]
    low_is_good : True for risk scores, False for health scores

    Returns
    -------
    HTML string safe to pass to st.markdown(..., unsafe_allow_html=True)
    """
    if low_is_good:
        color = (
            PALETTE["safe"] if value < 35
            else PALETTE["mid"]  if value < 65
            else PALETTE["risk"]
        )
    else:
        color = (
            PALETTE["risk"]  if value < 35
            else PALETTE["mid"]  if value < 65
            else PALETTE["safe"]
        )
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-fill" style="width:{value}%;background:{color};"></div>'
        f'</div>'
        f'<div class="bar-label">{value} / 100</div>'
    )
