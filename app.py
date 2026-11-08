"""
app.py - Supply Chain Analytics Inventory Decision Agent Application.

Dual-Mode Application:
1. Streamlit Web Dashboard: `streamlit run app.py`
2. Command-Line Interface (CLI): `python app.py "What is the EOQ for demand 10000, order cost $50, holding cost $2?"`
"""

import sys
import json
from typing import Dict, Any, Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from inventory_agent import InventoryDecisionAgent, ModelType, AgentResponse
import inventory_tools as tools


# ============================================================================
# VISUALIZATION BUILDERS (Plotly)
# ============================================================================

def build_eoq_plot(curve_data: Dict[str, Any], optimal_q: float, min_cost: float) -> go.Figure:
    fig = go.Figure()
    q_vals = curve_data["q_values"]

    fig.add_trace(go.Scatter(
        x=q_vals,
        y=curve_data["total_costs"],
        mode="lines",
        name="Total Inventory Cost",
        line=dict(color="#1f77b4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=q_vals,
        y=curve_data["holding_costs"],
        mode="lines",
        name="Annual Holding Cost (Q/2 * H)",
        line=dict(color="#2ca02c", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=q_vals,
        y=curve_data["ordering_costs"],
        mode="lines",
        name="Annual Ordering Cost (D/Q * S)",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
    ))

    # Mark optimal EOQ point
    fig.add_trace(go.Scatter(
        x=[optimal_q],
        y=[min_cost],
        mode="markers+text",
        name=f"Optimal EOQ: {optimal_q:,.1f}",
        text=[f"EOQ = {optimal_q:,.1f}<br>${min_cost:,.2f}"],
        textposition="top right",
        marker=dict(color="#d62728", size=12, symbol="star"),
    ))

    fig.update_layout(
        title="Economic Order Quantity (EOQ) Cost Trade-Off Curve",
        xaxis_title="Order Quantity (Q)",
        yaxis_title="Annual Cost ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_safety_stock_plot(curve_data: Dict[str, Any], mean_ltd: float, rop: float, csl: float) -> go.Figure:
    fig = go.Figure()
    x = curve_data["x"]
    pdf = curve_data["pdf"]

    fig.add_trace(go.Scatter(
        x=x,
        y=pdf,
        mode="lines",
        name="Lead Time Demand PDF",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.15)",
    ))

    # Add vertical line for Mean LTD
    fig.add_vline(
        x=mean_ltd,
        line_width=2,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"Mean LTD: {mean_ltd:,.1f}",
        annotation_position="top left",
    )

    # Add vertical line for ROP
    fig.add_vline(
        x=rop,
        line_width=2,
        line_dash="solid",
        line_color="#d62728",
        annotation_text=f"ROP: {rop:,.1f} ({csl*100:.0f}% CSL)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Lead Time Demand Distribution & Safety Stock Buffer",
        xaxis_title="Demand During Lead Time (units)",
        yaxis_title="Probability Density",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_newsvendor_plot(curve_data: Dict[str, Any], q_star: float, max_profit: float) -> go.Figure:
    fig = go.Figure()
    q_vals = curve_data["q_values"]
    profits = curve_data["profit_values"]

    fig.add_trace(go.Scatter(
        x=q_vals,
        y=profits,
        mode="lines",
        name="Expected Profit",
        line=dict(color="#2ca02c", width=3),
    ))

    # Mark optimal Q*
    fig.add_trace(go.Scatter(
        x=[q_star],
        y=[max_profit],
        mode="markers+text",
        name=f"Optimal Q*: {q_star:,.1f}",
        text=[f"Q* = {q_star:,.0f}<br>${max_profit:,.2f}"],
        textposition="top center",
        marker=dict(color="#d62728", size=12, symbol="star"),
    ))

    fig.update_layout(
        title="Newsvendor Expected Profit vs Order Quantity",
        xaxis_title="Order Quantity (Q)",
        yaxis_title="Expected Net Profit ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_quantity_discount_plot(results: Dict[str, Any]) -> go.Figure:
    tiers = results["tiers"]
    fig = go.Figure()

    tier_names = [f"Tier {t['tier_index']} (${t['unit_price']:.2f})" for t in tiers]
    purch_costs = [t["annual_purchase_cost"] for t in tiers]
    ord_costs = [t["annual_ordering_cost"] for t in tiers]
    hld_costs = [t["annual_holding_cost"] for t in tiers]

    fig.add_trace(go.Bar(name="Purchase Cost", x=tier_names, y=purch_costs, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Ordering Cost", x=tier_names, y=ord_costs, marker_color="#ff7f0e"))
    fig.add_trace(go.Bar(name="Holding Cost", x=tier_names, y=hld_costs, marker_color="#2ca02c"))

    fig.update_layout(
        barmode="stack",
        title="Annual Cost Breakdown by Discount Tier",
        xaxis_title="Discount Tiers",
        yaxis_title="Total Annual Cost ($)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# DATA ANALYSIS ENGINE (Pandas CSV Analytics)
# ============================================================================

def analyze_inventory_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates uploaded CSV and performs inventory dataset analytics using pandas:
    1. Validates required columns: Product, Category, Units, Unit Cost
    2. Calculates Total Units
    3. Calculates Average Unit Cost
    4. Identifies Product with Highest Units
    5. Groups data by Category and calculates total Units for each Category
    6. Creates calculated column: Inventory Value = Units * Unit Cost
    7. Calculates Total Inventory Value across all products
    """
    required_columns = ["Product", "Category", "Units", "Unit Cost"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "error_message": f"Missing required column(s): {', '.join(missing_columns)}",
        }

    df_analyzed = df.copy()

    # Clean/convert numeric columns safely if passed as formatted strings
    if not pd.api.types.is_numeric_dtype(df_analyzed["Units"]):
        df_analyzed["Units"] = pd.to_numeric(
            df_analyzed["Units"].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    else:
        df_analyzed["Units"] = pd.to_numeric(df_analyzed["Units"], errors="coerce")

    if not pd.api.types.is_numeric_dtype(df_analyzed["Unit Cost"]):
        df_analyzed["Unit Cost"] = pd.to_numeric(
            df_analyzed["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    else:
        df_analyzed["Unit Cost"] = pd.to_numeric(df_analyzed["Unit Cost"], errors="coerce")

    # 1. Calculate total Units
    total_units = float(df_analyzed["Units"].sum())

    # 2. Calculate average Unit Cost
    avg_unit_cost = float(df_analyzed["Unit Cost"].mean())

    # 3. Identify Product with highest Units
    if not df_analyzed.empty and df_analyzed["Units"].notnull().any():
        max_idx = df_analyzed["Units"].idxmax()
        highest_product = str(df_analyzed.loc[max_idx, "Product"])
        highest_units = float(df_analyzed.loc[max_idx, "Units"])
    else:
        highest_product = "N/A"
        highest_units = 0.0

    # 4. Group data by Category and calculate total Units for each Category
    category_summary = (
        df_analyzed.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
    )

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    df_analyzed["Inventory Value"] = df_analyzed["Units"] * df_analyzed["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = float(df_analyzed["Inventory Value"].sum())

    return {
        "valid": True,
        "dataframe": df_analyzed,
        "num_rows": len(df_analyzed),
        "num_columns": len(df_analyzed.columns),
        "column_names": list(df_analyzed.columns),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_product": highest_product,
        "highest_units": highest_units,
        "units_by_category": category_summary,
        "total_inventory_value": total_inventory_value,
    }


# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics: Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        """
        **Interactive Decision Support System for Inventory Management.**
        Built on the architecture:
        `Customer Question` ➔ `Identify Model` ➔ `Extract Inputs` ➔ `Select Python Tool` ➔ `Calculate` ➔ `Verify` ➔ `Explain`
        """
    )

    # Initialize agent in session state
    if "agent" not in st.session_state:
        st.session_state.agent = InventoryDecisionAgent()

    # Pre-canned Example Scenarios
    scenarios = {
        "Select a scenario or write your own...": "",
        "1. Economic Order Quantity (EOQ)": (
            "Our annual demand is 12,000 units, ordering cost is $60 per order, and annual holding cost is $4 per unit. "
            "Lead time is 5 days. What is our economic order quantity, total cost, and reorder point?"
        ),
        "2. Quantity Discount Schedule": (
            "Our annual demand is 5,000 units and ordering cost is $40 per order. Holding rate is 20% of unit price. "
            "Supplier offers discounts: 1-499 units at $20, 500-999 units at $18, 1000+ units at $15. "
            "Which quantity should we order to minimize total annual inventory cost?"
        ),
        "3. Safety Stock & Reorder Point (Continuous Review)": (
            "Daily demand is 50 units with a standard deviation of 8 units. Supplier lead time is 5 days. "
            "We want a 95% cycle service level. Calculate our safety stock and reorder point."
        ),
        "4. Fixed-Period Review (Order-Up-To Policy)": (
            "We review inventory every 14 days with a lead time of 4 days. Daily demand averages 25 units with std dev of 5 units. "
            "We target a 98% service level. Current on-hand inventory is 120 units and 40 units are currently on order. "
            "What is our order-up-to target level and how much should we order today?"
        ),
        "5. Single-Period Newsvendor (Perishable / Seasonal)": (
            "We sell a seasonal fashion ski jacket for $180 per unit that costs $70 to produce. "
            "Unsold jackets at season end are salvaged for $30 each. Expected demand is 500 jackets with standard deviation of 80. "
            "How many jackets should we order to maximize expected profit?"
        ),
    }

    # Sidebar
    with st.sidebar:
        st.header("🎯 Preset Scenarios")
        selected_scenario = st.selectbox("Load an example problem:", list(scenarios.keys()))
        st.markdown("---")
        st.markdown(
            """
            ### 📚 Supported Models
            1. **EOQ (Economic Order Quantity)**
            2. **Quantity Discount Optimization**
            3. **Safety Stock & Reorder Point (ROP)**
            4. **Fixed-Period Review Policy**
            5. **Newsvendor Model**
            """
        )
        st.markdown("---")
        st.info("💡 **Tip:** You can ask questions in natural language. The agent automatically detects the inventory model, extracts the parameters, runs verified numerical calculations, and generates actionable advice.")
        st.markdown("---")
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # ------------------------------------------------------------------------
    # CSV DATA UPLOAD & ANALYSIS SECTION
    # ------------------------------------------------------------------------
    if uploaded_file is not None:
        st.markdown("---")
        st.header("📊 Inventory Data Analysis")
        try:
            raw_df = pd.read_csv(uploaded_file)
            analysis = analyze_inventory_dataframe(raw_df)

            if not analysis["valid"]:
                st.error(
                    f"❌ Missing required columns in uploaded CSV: {', '.join(analysis['missing_columns'])}. "
                    "The CSV file must include: Product, Category, Units, Unit Cost."
                )
            else:
                # 1. Dataset Dimensions & Column Info
                st.subheader("📋 Dataset Overview")
                d_col1, d_col2, d_col3 = st.columns(3)
                with d_col1:
                    st.metric("Number of Rows", f"{analysis['num_rows']:,}")
                with d_col2:
                    st.metric("Number of Columns", f"{analysis['num_columns']:,}")
                with d_col3:
                    st.write(f"**Column Names:** {', '.join(analysis['column_names'])}")

                # 2. Key Analysis Results Metrics
                st.subheader("📈 Inventory Analysis Results")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    tot_units = analysis["total_units"]
                    st.metric("Total Units", f"{int(tot_units):,}" if tot_units.is_integer() else f"{tot_units:,.2f}")
                with m2:
                    st.metric("Average Unit Cost", f"${analysis['avg_unit_cost']:,.2f}")
                with m3:
                    highest_u = analysis["highest_units"]
                    u_str = f"{int(highest_u):,}" if highest_u.is_integer() else f"{highest_u:,.2f}"
                    st.metric(
                        "Product with Highest Units",
                        analysis["highest_product"],
                        f"{u_str} units",
                    )
                with m4:
                    st.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

                # 3. Total Units by Category
                st.subheader("🏷️ Total Units by Category")
                cat_col1, cat_col2 = st.columns([1, 1])
                with cat_col1:
                    formatted_cat = analysis["units_by_category"].copy()
                    formatted_cat["Total Units"] = formatted_cat["Total Units"].apply(
                        lambda x: f"{int(x):,}" if isinstance(x, (int, float)) and float(x).is_integer() else f"{x:,.2f}"
                    )
                    st.dataframe(formatted_cat, use_container_width=True)
                with cat_col2:
                    fig_cat = px.bar(
                        analysis["units_by_category"],
                        x="Category",
                        y="Total Units",
                        color="Category",
                        title="Total Units by Category",
                    )
                    fig_cat.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_cat, use_container_width=True)

                # 4. Display Dataframe including the new Inventory Value column
                st.subheader("📦 Inventory Data (including Inventory Value)")
                display_df = analysis["dataframe"].copy()
                st.dataframe(
                    display_df.style.format({
                        "Unit Cost": "${:,.2f}",
                        "Inventory Value": "${:,.2f}",
                        "Units": "{:,.0f}",
                    }),
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    # Input text area
    default_text = scenarios[selected_scenario] if selected_scenario != "Select a scenario or write your own..." else ""
    user_query = st.text_area(
        "💬 Enter your inventory question in natural language:",
        value=default_text,
        height=120,
        placeholder="e.g. Our annual demand is 10,000 units, order cost is $50, holding cost is $2. What is our EOQ?",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze & Decide", type="primary")

    if analyze_clicked and user_query.strip():
        with st.spinner("Processing inquiry through Supply Chain Decision Pipeline..."):
            response: AgentResponse = st.session_state.agent.run(user_query)
            st.session_state.last_response = response

    if "last_response" in st.session_state:
        resp: AgentResponse = st.session_state.last_response

        st.markdown("---")
        st.subheader("🔍 Decision Agent Pipeline Execution")

        # 6-Step Pipeline Visualizer
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f"**1. Model Identified**\n\n`{resp.identified_model.value}`\n\n*(Conf: {resp.model_confidence*100:.0f}%)*")
        with c2:
            st.markdown(f"**2. Inputs Extracted**\n\n`{len(resp.extracted_inputs)} fields`\n\n*Auto-parsed*")
        with c3:
            st.markdown(f"**3. Tool Selected**\n\n`{resp.tool_selected}`\n\n*inventory_tools.py*")
        with c4:
            calc_icon = "✅ Success" if resp.calculation_results else "❌ Failed"
            st.markdown(f"**4. Calculation**\n\n`{calc_icon}`\n\n*Deterministic*")
        with c5:
            badge = "✅ Passed" if resp.verification_overall == "PASSED" else ("⚠️ Warning" if resp.verification_overall == "WARNING" else "❌ Failed")
            st.markdown(f"**5. Verification**\n\n`{badge}`\n\n*{len(resp.verification_checks)} rules checked*")
        with c6:
            st.markdown("**6. Explanation**\n\n`Generated`\n\n*Executive Advisory*")

        st.markdown("---")

        # Tabs for Advisory, Interactive Tuning, Visuals, and Execution Trace
        tab_advisory, tab_visuals, tab_tuning, tab_trace = st.tabs([
            "📋 Supply Chain Advisory",
            "📊 Analytical Visualizations",
            "🎛️ What-If Parameter Fine-Tuning",
            "🔬 Pipeline Architecture Trace",
        ])

        with tab_advisory:
            st.markdown(resp.explanation_markdown)

        with tab_visuals:
            if resp.calculation_results:
                if resp.identified_model == ModelType.EOQ and resp.curve_data:
                    fig = build_eoq_plot(
                        resp.curve_data,
                        resp.calculation_results["optimal_order_quantity"],
                        resp.calculation_results["total_annual_inventory_cost"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif resp.identified_model == ModelType.QUANTITY_DISCOUNT:
                    fig = build_quantity_discount_plot(resp.calculation_results)
                    st.plotly_chart(fig, use_container_width=True)

                elif resp.identified_model == ModelType.SAFETY_STOCK_ROP and resp.curve_data:
                    fig = build_safety_stock_plot(
                        resp.curve_data,
                        resp.calculation_results["expected_lead_time_demand"],
                        resp.calculation_results["reorder_point"],
                        resp.calculation_results["service_level"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif resp.identified_model == ModelType.NEWSEVNDOR and resp.curve_data:
                    fig = build_newsvendor_plot(
                        resp.curve_data,
                        resp.calculation_results["optimal_order_quantity"],
                        resp.calculation_results["expected_profit"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif resp.identified_model == ModelType.FIXED_PERIOD_REVIEW:
                    st.info("Periodic Review Sawtooth Visualization")
                    res = resp.calculation_results
                    df = pd.DataFrame({
                        "Metric": ["Order-Up-To Level (M)", "Inventory Position (On-Hand + On-Order)", "Recommended Order Quantity (Q)", "Safety Stock (SS)"],
                        "Units": [res["order_up_to_level"], res["inventory_position"], res["recommended_order_quantity"], res["safety_stock"]],
                    })
                    fig = px.bar(df, x="Metric", y="Units", color="Metric", title="Periodic Review Target Inventory Breakdown")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Visualizations require complete calculation results.")

        with tab_tuning:
            st.markdown("#### Dynamic Sensitivity & Fine-Tuning")
            st.markdown("Adjust the extracted parameters below to immediately recalculate recommendations:")

            overrides = {}
            if resp.extracted_inputs:
                cols = st.columns(min(3, len(resp.extracted_inputs)))
                for idx, (k, v) in enumerate(resp.extracted_inputs.items()):
                    if isinstance(v, (int, float)):
                        col = cols[idx % len(cols)]
                        with col:
                            step = 1.0 if v >= 10 else 0.05
                            min_val = 0.0 if "rate" in k or "cost" in k or "inventory" in k else 1.0
                            new_v = st.number_input(f"`{k}`", value=float(v), min_value=min_val, step=step)
                            overrides[k] = new_v

                if st.button("🔄 Recalculate with Fine-Tuned Inputs"):
                    updated_resp = st.session_state.agent.run(resp.question, override_inputs=overrides)
                    st.session_state.last_response = updated_resp
                    st.rerun()

        with tab_trace:
            st.markdown("#### Detailed Execution Trace")
            for t in resp.execution_trace:
                with st.expander(f"{t.step_name} — Status: {t.status}"):
                    st.json(t.details)

            st.markdown("#### Full Agent Response Object")
            st.json(resp.to_dict())


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_cli(query: str):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    agent = InventoryDecisionAgent()
    print("\n" + "=" * 80)
    print("SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT")
    print("=" * 80)
    print(f"\n[Customer Question]:\n{query}\n")

    response = agent.run(query)

    print("-" * 80)
    print("PIPELINE EXECUTION TRACE:")
    for step in response.execution_trace:
        print(f"  * {step.step_name}: [{step.status}] {step.details}")
    print("-" * 80)

    print("\n" + response.explanation_markdown + "\n")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Run CLI with argument question
        cli_question = " ".join(sys.argv[1:])
        run_cli(cli_question)
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        agent = InventoryDecisionAgent()
        print("\nEntering Supply Chain Inventory Agent interactive mode (type 'exit' to quit):\n")
        while True:
            try:
                q = input("\nAsk an inventory question > ")
                if q.strip().lower() in ["exit", "quit", "q"]:
                    break
                if not q.strip():
                    continue
                run_cli(q)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Default to Streamlit runner
        run_streamlit_app()
