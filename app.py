"""Supply Chain Analytics Inventory Decision Agent Application.

Dual-mode interface:
1. Interactive Streamlit Web Dashboard: `streamlit run app.py`
2. Command-Line Interface (CLI): `python app.py --cli` or `python app.py "your question here"`

Architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from inventory_agent import (
    MODEL_DISPLAY_NAMES,
    AgentPipelineTrace,
    InventoryDecisionAgent,
    ModelType,
)
from inventory_tools import (
    EOQResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
    PeriodicReviewResult,
    QuantityDiscountResult,
    ROPResult,
    SafetyStockROPResult,
    calculate_eoq,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
)

# -----------------------------------------------------------------------------
# Pre-loaded Example Scenarios
# -----------------------------------------------------------------------------

EXAMPLE_QUESTIONS = {
    "1. Economic Order Quantity (EOQ)": (
        "Our annual demand is 12,000 units. Ordering setup cost is $50 per order, "
        "and holding cost is $4 per unit per year. Unit cost is $25. "
        "What is our optimal order quantity and how often should we place orders?"
    ),
    "2. Quantity Discount Analysis": (
        "We have annual demand of 10,000 units and ordering cost is $50 per order. "
        "Holding rate is 20% of price. Supplier offers tiers: "
        "1-499 units at $10.00, 500-999 units at $9.00, 1000+ units at $8.00. "
        "Which discount bracket should we order?"
    ),
    "3. Safety Stock & Reorder Point": (
        "Average daily demand is 80 units with a standard deviation of 15 units. "
        "Supplier lead time is 5 days. Unit cost is $45 and holding rate is 20%. "
        "What safety stock and reorder point should we set for a 95% cycle service level?"
    ),
    "4. Fixed-Period Review (Order-Up-To)": (
        "We do a periodic inventory review every 7 days. Supplier lead time is 3 days. "
        "Average daily demand is 50 units with standard deviation of 10 units. "
        "Target service level is 95%. Currently we have 120 units on hand, 50 units on order, "
        "and 0 backorders. What is our order-up-to target and how many units should we order today?"
    ),
    "5. Newsvendor (Seasonal / Perishable)": (
        "We are ordering designer ski jackets for the upcoming winter season. "
        "Retail selling price is $180, wholesale purchase cost is $75, and unsold jackets "
        "can be salvaged for $30. Demand is normally distributed with mean 650 and std dev 120. "
        "What is our optimal stocking quantity, critical ratio, and expected profit?"
    ),
}


# -----------------------------------------------------------------------------
# Plotly Chart Generators
# -----------------------------------------------------------------------------

def build_eoq_chart(res: EOQResult) -> go.Figure:
    """Build interactive EOQ trade-off curve figure."""
    fig = go.Figure()
    curve = res.cost_curve
    q_vals = curve.get("order_quantities", [])
    hold_costs = curve.get("holding_costs", [])
    ord_costs = curve.get("ordering_costs", [])
    tot_costs = curve.get("total_inventory_costs", [])

    fig.add_trace(go.Scatter(
        x=q_vals, y=hold_costs, mode="lines", name="Annual Holding Cost",
        line=dict(color="#2ca02c", dash="dot", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=q_vals, y=ord_costs, mode="lines", name="Annual Ordering Cost",
        line=dict(color="#1f77b4", dash="dash", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=q_vals, y=tot_costs, mode="lines", name="Total Inventory Cost",
        line=dict(color="#d62728", width=3)
    ))

    # Add vertical line at EOQ
    fig.add_vline(
        x=res.eoq, line_width=2, line_dash="dash", line_color="#9467bd",
        annotation_text=f"EOQ = {res.eoq:.0f}", annotation_position="top right"
    )

    fig.update_layout(
        title="EOQ Cost Trade-off: Holding vs. Ordering vs. Total Cost",
        xaxis_title="Order Quantity (Q)",
        yaxis_title="Annual Cost ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_discount_chart(res: QuantityDiscountResult) -> go.Figure:
    """Build comparison chart across price break tiers."""
    fig = go.Figure()
    tiers = res.tier_evaluations
    labels = [f"Tier {t.tier_index + 1}<br>(${t.unit_price:.2f})" for t in tiers]
    purch_costs = [t.annual_purchase_cost for t in tiers]
    ord_costs = [t.annual_ordering_cost for t in tiers]
    hold_costs = [t.annual_holding_cost for t in tiers]
    tot_costs = [t.total_annual_cost for t in tiers]

    fig.add_trace(go.Bar(name="Purchase Cost", x=labels, y=purch_costs, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Ordering Cost", x=labels, y=ord_costs, marker_color="#ff7f0e"))
    fig.add_trace(go.Bar(name="Holding Cost", x=labels, y=hold_costs, marker_color="#2ca02c"))

    # Highlight optimal tier
    colors = ["#d62728" if t.tier_index == res.optimal_tier_index else "#7f7f7f" for t in tiers]
    fig.add_trace(go.Scatter(
        name="Grand Total Cost", x=labels, y=tot_costs, mode="lines+markers+text",
        text=[f"${tc:,.0f}" for tc in tot_costs], textposition="top center",
        line=dict(color="#d62728", width=2), marker=dict(size=12, color=colors)
    ))

    fig.update_layout(
        barmode="stack",
        title="Total Annual Cost Breakdown Across Discount Tiers",
        xaxis_title="Price Tiers",
        yaxis_title="Annual Cost ($)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_safety_stock_chart(res: SafetyStockROPResult) -> go.Figure:
    """Build service level vs safety stock & holding cost curve."""
    tradeoffs = res.service_level_tradeoffs
    if not tradeoffs:
        return go.Figure()

    df = pd.DataFrame(tradeoffs)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[f"{sl:.1%}" for sl in df["service_level"]],
        y=df["safety_stock"],
        mode="lines+markers",
        name="Safety Stock (Units)",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8),
        yaxis="y1"
    ))

    if res.unit_cost > 0:
        fig.add_trace(go.Scatter(
            x=[f"{sl:.1%}" for sl in df["service_level"]],
            y=df["annual_ss_holding_cost"],
            mode="lines+markers",
            name="Holding Cost ($/yr)",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            marker=dict(size=8),
            yaxis="y2"
        ))

    fig.update_layout(
        title="Service Level vs. Required Safety Stock & Holding Investment",
        xaxis_title="Cycle Service Level (CSL)",
        yaxis=dict(title="Safety Stock (Units)", side="left"),
        yaxis2=dict(title="Annual Holding Cost ($)", side="right", overlaying="y", showgrid=False) if res.unit_cost > 0 else None,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_fixed_period_chart(res: FixedPeriodReviewResult) -> go.Figure:
    """Build inventory position vs target stock level waterfall/bar chart."""
    fig = go.Figure()

    categories = [
        "On Hand",
        "On Order",
        "Backorders",
        "Effective Inv Position",
        "Recommended Order",
        "Target Stock Level (S)"
    ]
    values = [
        res.current_on_hand,
        res.inventory_on_order,
        -res.backorders,
        res.inventory_position,
        res.recommended_order_quantity,
        res.recommended_target_level
    ]
    colors = ["#1f77b4", "#17becf", "#e377c2", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Periodic Review Replenishment: Target Stock ({res.recommended_target_level:,}) vs Position ({res.inventory_position:,.0f})",
        yaxis_title="Units",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def build_newsvendor_chart(res: NewsvendorResult) -> go.Figure:
    """Build normal distribution curve showing optimal stocking cutoff."""
    mu = res.mean_demand
    sigma = res.std_demand
    q_opt = res.optimal_order_quantity

    x_vals = np.linspace(max(0, mu - 3.5 * sigma), mu + 3.5 * sigma, 300)
    pdf_vals = stats.norm.pdf(x_vals, mu, sigma)

    fig = go.Figure()

    # Full distribution curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=pdf_vals, mode="lines", name="Demand Distribution N(μ, σ)",
        line=dict(color="#1f77b4", width=2)
    ))

    # In-stock shaded area (demand <= Q*)
    x_instock = x_vals[x_vals <= q_opt]
    y_instock = pdf_vals[x_vals <= q_opt]
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_instock, [q_opt]]),
        y=np.concatenate([y_instock, [0]]),
        fill="tozeroy",
        fillcolor="rgba(44, 160, 44, 0.35)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"In-Stock Probability ({res.critical_ratio:.1%})"
    ))

    # Vertical line at Q*
    fig.add_vline(
        x=q_opt, line_width=2.5, line_dash="dash", line_color="#d62728",
        annotation_text=f"Q* = {res.recommended_order_quantity:,}", annotation_position="top right"
    )

    fig.update_layout(
        title=f"Newsvendor Demand Curve: Optimal Q* = {res.recommended_order_quantity:,} (Critical Ratio = {res.critical_ratio:.1%})",
        xaxis_title="Demand Quantity",
        yaxis_title="Probability Density",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# Streamlit Web Interface
# -----------------------------------------------------------------------------

def render_streamlit_app():
    """Render the full Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics: Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-left: 5px solid #1f77b4;
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 12px;
        }
        .pipeline-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            background-color: #e3f2fd;
            color: #0d47a1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize agent in session state
    if "agent" not in st.session_state:
        st.session_state.agent = InventoryDecisionAgent()

    # Header
    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "**Autonomous Decision Engine for Inventory Planning & Optimization**  \n"
        "Architecture: `Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain`"
    )

    # Sidebar: Model Directory & Pre-loaded Scenarios
    with st.sidebar:
        st.header("🎯 Inventory Models")
        st.markdown(
            """
            1. **Economic Order Quantity (EOQ)**
            2. **Quantity Discount Analysis**
            3. **Safety Stock & Reorder Point (ROP)**
            4. **Fixed-Period Review Policy (R, S)**
            5. **Newsvendor Single-Period Model**
            """
        )
        st.divider()

        st.subheader("💡 Quick-Load Scenarios")
        selected_scenario = st.selectbox(
            "Load a pre-configured scenario:",
            ["(Custom Question)"] + list(EXAMPLE_QUESTIONS.keys()),
        )

        model_override = st.selectbox(
            "Model Classification Override:",
            ["Auto-Detect"] + [MODEL_DISPLAY_NAMES[m] for m in ModelType if m != ModelType.UNKNOWN],
            index=0,
        )

        st.divider()
        st.info(
            "⚡ **Deterministic Guarantee**: All mathematical equations, statistics, and bounds "
            "are calculated using audited Python code (Scipy & NumPy), guaranteeing zero LLM calculation hallucination."
        )

    # Populate question text based on scenario
    default_text = ""
    if selected_scenario != "(Custom Question)":
        default_text = EXAMPLE_QUESTIONS[selected_scenario]

    # Question Input
    st.subheader("🗣️ Natural-Language Inventory Question")
    user_question = st.text_area(
        "Enter your supply chain inventory question or requirement:",
        value=default_text,
        height=100,
        placeholder="e.g. Annual demand is 12000 units, order setup cost is $50, and holding cost is $4 per unit per year. What is our optimal lot size and how often should we order?",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_analysis = st.button("🚀 Analyze Decision", type="primary", use_container_width=True)

    if run_analysis or default_text:
        if not user_question.strip():
            st.warning("Please enter a question or select a scenario above.")
            return

        override_enum: Optional[ModelType] = None
        if model_override != "Auto-Detect":
            for m, name in MODEL_DISPLAY_NAMES.items():
                if name == model_override:
                    override_enum = m
                    break

        with st.spinner("Executing 7-stage Supply Chain Decision Pipeline..."):
            trace: AgentPipelineTrace = st.session_state.agent.answer_question(
                user_question, override_model=override_enum
            )

        st.success("Decision Pipeline Completed Successfully!")

        # ---------------------------------------------------------------------
        # TOP EXECUTIVE SUMMARY & KEY METRICS
        # ---------------------------------------------------------------------
        st.subheader("📋 Executive Decision Summary")
        st.info(trace.explanation.executive_summary)

        # High-level metric columns
        if trace.calculation and trace.verification.is_valid:
            res_obj = trace.calculation.result_object
            m_cols = st.columns(4)

            if trace.model_identification.model_type == ModelType.EOQ:
                m_cols[0].metric("Recommended Order Qty", f"{res_obj.recommended_order_quantity:,} units")
                m_cols[1].metric("Orders Per Year", f"{res_obj.orders_per_year:.1f}")
                m_cols[2].metric("Cycle Interval", f"{res_obj.order_cycle_days:.1f} days")
                m_cols[3].metric("Total Annual Inv Cost", f"${res_obj.total_annual_inventory_cost:,.2f}")

            elif trace.model_identification.model_type == ModelType.QUANTITY_DISCOUNT:
                m_cols[0].metric("Optimal Order Qty", f"{res_obj.recommended_order_quantity:,} units")
                m_cols[1].metric("Unit Price", f"${res_obj.optimal_unit_price:,.2f}")
                m_cols[2].metric("Annual Cost Savings", f"${res_obj.annual_cost_savings:,.2f}")
                m_cols[3].metric("Total Annual Cost", f"${res_obj.optimal_total_annual_cost:,.2f}")

            elif trace.model_identification.model_type == ModelType.SAFETY_STOCK_ROP:
                m_cols[0].metric("Reorder Point (ROP)", f"{res_obj.recommended_reorder_point:,} units")
                m_cols[1].metric("Safety Stock Buffer", f"{res_obj.recommended_safety_stock:,} units")
                m_cols[2].metric("Cycle Service Level", f"{res_obj.service_level:.1%}")
                m_cols[3].metric("Stockout Risk / Cycle", f"{res_obj.stockout_risk_per_cycle:.1%}")

            elif trace.model_identification.model_type == ModelType.FIXED_PERIOD_REVIEW:
                m_cols[0].metric("Target Stock Level (S)", f"{res_obj.recommended_target_level:,} units")
                m_cols[1].metric("Order Quantity to Place", f"{res_obj.recommended_order_quantity:,} units")
                m_cols[2].metric("Current Position (IP)", f"{res_obj.inventory_position:,.0f} units")
                m_cols[3].metric("Protection Window (R+L)", f"{res_obj.protection_interval_days:.1f} days")

            elif trace.model_identification.model_type == ModelType.NEWSVENDOR:
                m_cols[0].metric("Optimal Stocking (Q*)", f"{res_obj.recommended_order_quantity:,} units")
                m_cols[1].metric("Critical Fractile (CR)", f"{res_obj.critical_ratio:.1%}")
                m_cols[2].metric("Expected Net Profit", f"${res_obj.expected_profit:,.2f}")
                m_cols[3].metric("Expected Lost Sales", f"{res_obj.expected_lost_sales:,.1f} units")

        # ---------------------------------------------------------------------
        # INTERACTIVE CHART VISUALIZATION
        # ---------------------------------------------------------------------
        if trace.calculation and trace.verification.is_valid:
            st.subheader("📊 Interactive Analytical Visualizations")
            mt = trace.model_identification.model_type
            if mt == ModelType.EOQ:
                st.plotly_chart(build_eoq_chart(trace.calculation.result_object), use_container_width=True)
            elif mt == ModelType.QUANTITY_DISCOUNT:
                st.plotly_chart(build_discount_chart(trace.calculation.result_object), use_container_width=True)
            elif mt == ModelType.SAFETY_STOCK_ROP:
                st.plotly_chart(build_safety_stock_chart(trace.calculation.result_object), use_container_width=True)
            elif mt == ModelType.FIXED_PERIOD_REVIEW:
                st.plotly_chart(build_fixed_period_chart(trace.calculation.result_object), use_container_width=True)
            elif mt == ModelType.NEWSVENDOR:
                st.plotly_chart(build_newsvendor_chart(trace.calculation.result_object), use_container_width=True)

        # ---------------------------------------------------------------------
        # DETAILED EXPLANATION & AUDIT REPORT
        # ---------------------------------------------------------------------
        st.subheader("📑 Formal Analytical Report")
        st.markdown(trace.explanation.markdown_report)

        # ---------------------------------------------------------------------
        # PIPELINE TRACE INSPECTOR (7-STAGE ACCORDION)
        # ---------------------------------------------------------------------
        st.subheader("🔍 Architectural Pipeline Inspector (Audit Trail)")
        with st.expander("Inspect Step-by-Step Execution Pipeline (Stages 1-7)", expanded=False):
            t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                "1. Question", "2. Model Identification", "3. Extracted Inputs",
                "4. Selected Tool", "5. Calculation Output", "6. Verification", "7. Explanation"
            ])

            with t1:
                st.markdown(f"**Customer Raw Input:**\n> {trace.question}")

            with t2:
                mid = trace.model_identification
                st.markdown(f"**Identified Model:** `{mid.model_name}` ({mid.model_type.value})")
                st.markdown(f"**Classification Confidence:** `{mid.confidence:.1%}`")
                st.markdown(f"**Classification Rationale:** {mid.rationale}")
                if mid.alternative_candidates:
                    st.markdown("**Alternative Candidates:**")
                    for name, score in mid.alternative_candidates:
                        st.markdown(f"- {name}: score `{score}`")

            with t3:
                inp = trace.extracted_inputs
                st.markdown("**Extracted Parameters:**")
                st.json(inp.parameters)
                if inp.defaults_applied:
                    st.markdown("**Defaults Injected:**")
                    st.json(inp.defaults_applied)
                if inp.notes:
                    st.markdown("**Extraction Notes:**")
                    for n in inp.notes:
                        st.markdown(f"- {n}")
                if inp.missing_critical:
                    st.error(f"Missing Critical Parameters: {', '.join(inp.missing_critical)}")

            with t4:
                ts = trace.tool_selection
                st.markdown(f"**Python Function Invoked:** `{ts.tool_function}`")
                st.markdown("**Mapped Arguments:**")
                st.json(ts.arguments)

            with t5:
                if trace.calculation:
                    st.markdown("**Deterministic Numerical Results:**")
                    st.json(trace.calculation.result_dict)
                else:
                    st.warning("Calculation did not run due to missing inputs.")

            with t6:
                v = trace.verification
                if v.is_valid:
                    st.success("✅ All Invariant Checks Passed")
                else:
                    st.error("❌ Verification Failed")
                st.markdown("**Passed Invariants:**")
                for c in v.checks_passed:
                    st.markdown(f"- ✅ {c}")
                if v.warnings:
                    st.markdown("**Warnings:**")
                    for w in v.warnings:
                        st.markdown(f"- ⚠️ {w}")
                if v.errors:
                    st.markdown("**Errors:**")
                    for e in v.errors:
                        st.markdown(f"- ❌ {e}")

            with t7:
                st.markdown("**Executive Recommendations:**")
                for r in trace.explanation.actionable_recommendations:
                    st.markdown(f"- 📌 {r}")


# -----------------------------------------------------------------------------
# Command Line Interface (CLI)
# -----------------------------------------------------------------------------

def run_cli_interactive():
    """Run an interactive CLI session."""
    agent = InventoryDecisionAgent()

    print("=" * 80)
    print("📦 SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT (CLI MODE)")
    print("=" * 80)
    print("Available Inventory Models:")
    print("  1. Economic Order Quantity (EOQ)")
    print("  2. Quantity Discount Analysis")
    print("  3. Safety Stock & Reorder Point (SS & ROP)")
    print("  4. Fixed-Period Review (Order-Up-To Level)")
    print("  5. Newsvendor Single-Period Stochastic Model")
    print("Type your question below, or type 'samples' to see examples, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\n[Enter Question] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting. Goodbye!")
            break
        if user_input.lower() in ["samples", "examples", "help"]:
            print("\nPre-configured Example Questions:")
            for name, q in EXAMPLE_QUESTIONS.items():
                print(f"\n--- {name} ---")
                print(q)
            continue

        trace = agent.answer_question(user_input)
        print_cli_trace(trace)


def print_cli_trace(trace: AgentPipelineTrace):
    """Print structured pipeline execution results to console."""
    print("\n" + "=" * 80)
    print(f"STAGE 1: QUESTION: {trace.question}")
    print("-" * 80)
    print(f"STAGE 2: IDENTIFIED MODEL: {trace.model_identification.model_name} (Confidence: {trace.model_identification.confidence:.1%})")
    print(f"         Rationale: {trace.model_identification.rationale}")
    print("-" * 80)
    print(f"STAGE 3: EXTRACTED INPUTS: {json.dumps(trace.extracted_inputs.parameters, indent=2)}")
    if trace.extracted_inputs.defaults_applied:
        print(f"         Defaults Applied: {trace.extracted_inputs.defaults_applied}")
    print("-" * 80)
    print(f"STAGE 4: SELECTED PYTHON TOOL: {trace.tool_selection.tool_function}")
    print("-" * 80)
    print(f"STAGE 5: CALCULATION STATUS: {'SUCCESS' if trace.calculation else 'FAILED'}")
    print("-" * 80)
    print(f"STAGE 6: VERIFICATION: {'VALID' if trace.verification.is_valid else 'INVALID'}")
    for c in trace.verification.checks_passed:
        print(f"         [PASSED] {c}")
    for w in trace.verification.warnings:
        print(f"         [WARNING] {w}")
    for e in trace.verification.errors:
        print(f"         [ERROR] {e}")
    print("-" * 80)
    print("STAGE 7: EXPLANATION:")
    print(f"\n{trace.explanation.executive_summary}\n")
    print("Actionable Recommendations:")
    for rec in trace.explanation.actionable_recommendations:
        print(f"  • {rec}")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Main Application Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("question", nargs="*", help="Natural-language question to evaluate")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI session")
    args, unknown = parser.parse_known_args()

    # If invoked directly via `streamlit run app.py`
    is_streamlit_runtime = False
    try:
        import streamlit.runtime as st_rt
        is_streamlit_runtime = st_rt.exists()
    except Exception:
        pass

    if is_streamlit_runtime:
        render_streamlit_app()
    elif args.cli:
        run_cli_interactive()
    elif args.question:
        question_str = " ".join(args.question)
        agent = InventoryDecisionAgent()
        trace = agent.answer_question(question_str)
        print_cli_trace(trace)
    else:
        # Check if user invoked `python app.py` directly without arguments
        print("Supply Chain Analytics: Inventory Decision Agent")
        print("To launch the Streamlit Web Dashboard: streamlit run app.py")
        print("To run in interactive CLI mode:        python app.py --cli")
        print("To run a single query:                 python app.py \"Your inventory question here\"\n")
        print("Starting interactive CLI session now...\n")
        run_cli_interactive()


if __name__ == "__main__":
    main()
