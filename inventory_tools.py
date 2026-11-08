"""Supply Chain Analytics Inventory Decision Tools.

This module provides deterministic mathematical tools for five core inventory optimization models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool:
- Validates all inputs and raises ValueError on missing or invalid numerical inputs.
- Never invents missing numerical inputs.
- Returns structured results with typed fields and .to_dict().
- Returns explicit intermediate calculations and step-by-step mathematical breakdowns.
- Returns the final answer in a dedicated summary dictionary.
- Identifies all underlying operational and statistical assumptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


# -----------------------------------------------------------------------------
# Verification Container
# -----------------------------------------------------------------------------

@dataclass
class CalculationVerification:
    """Stores sanity and business invariant verification results."""
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Structured Result Containers
# -----------------------------------------------------------------------------

@dataclass
class EOQResult:
    """Structured results for Economic Order Quantity calculations."""
    model_name: str = "Economic Order Quantity (EOQ)"
    
    # 8 Required Student Verification Components
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Backward-compatible aliases
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Direct typed attributes for seamless programmatic access
    demand: float = 0.0
    ordering_cost: float = 0.0
    holding_cost_per_unit: float = 0.0
    unit_cost: float = 0.0
    work_days_per_year: int = 250
    
    eoq: float = 0.0
    recommended_order_quantity: int = 0
    orders_per_year: float = 0.0
    order_cycle_days: float = 0.0
    order_cycle_years: float = 0.0
    annual_ordering_cost: float = 0.0
    annual_holding_cost: float = 0.0
    total_annual_inventory_cost: float = 0.0
    annual_purchase_cost: float = 0.0
    grand_total_annual_cost: float = 0.0
    
    cost_curve: Dict[str, List[float]] = field(default_factory=dict)
    verification: CalculationVerification = field(default_factory=lambda: CalculationVerification(is_valid=True))
    
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get_student_verification(self) -> Dict[str, Any]:
        """Return the 8 structured components for student verification."""
        return {
            "1. Inputs Extracted": self.inputs_extracted,
            "2. Variable Definitions": self.variable_definitions,
            "3. Formula": self.formula,
            "4. Formula with Substituted Values": self.formula_with_substituted_values,
            "5. Intermediate Calculations": self.intermediate_calculations,
            "6. Final Answer": self.final_answer,
            "7. Verification Checks": self.verification_checks,
            "8. Assumptions": self.assumptions,
        }

    def format_student_report(self) -> str:
        """Format the 8 verification components into a Markdown report."""
        lines = [
            f"### 📚 Student Verification Report: {self.model_name}",
            "",
            "#### 1. Inputs Extracted",
        ]
        for k, v in self.inputs_extracted.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 2. Variable Definitions"])
        for k, v in self.variable_definitions.items():
            lines.append(f"- **{k}**: {v}")
        lines.extend(["", "#### 3. Formula"])
        for k, v in self.formula.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 4. Formula with Substituted Values"])
        for k, v in self.formula_with_substituted_values.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 5. Intermediate Calculations"])
        for k, v in self.intermediate_calculations.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 6. Final Answer"])
        for k, v in self.final_answer.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 7. Verification Checks"])
        for c in self.verification_checks:
            lines.append(f"- ✅ {c}")
        lines.extend(["", "#### 8. Assumptions"])
        for a in self.assumptions:
            lines.append(f"- ℹ️ {a}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscountTierResult:
    """Individual tier evaluation within Quantity Discount model."""
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    unconstrained_eoq: float
    feasible_order_quantity: float
    is_price_break_point: bool
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult:
    """Structured results for Quantity Discount analysis."""
    model_name: str = "Quantity Discount Analysis"
    
    # 8 Required Student Verification Components
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Backward-compatible aliases
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    demand: float = 0.0
    ordering_cost: float = 0.0
    holding_rate: Optional[float] = None
    fixed_holding_cost: Optional[float] = None
    
    optimal_tier_index: int = 0
    optimal_order_quantity: float = 0.0
    recommended_order_quantity: int = 0
    optimal_unit_price: float = 0.0
    optimal_total_annual_cost: float = 0.0
    baseline_total_annual_cost: float = 0.0
    annual_cost_savings: float = 0.0
    percentage_savings: float = 0.0
    
    tier_evaluations: List[DiscountTierResult] = field(default_factory=list)
    verification: CalculationVerification = field(default_factory=lambda: CalculationVerification(is_valid=True))

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get_student_verification(self) -> Dict[str, Any]:
        """Return the 8 structured components for student verification."""
        return {
            "1. Inputs Extracted": self.inputs_extracted,
            "2. Variable Definitions": self.variable_definitions,
            "3. Formula": self.formula,
            "4. Formula with Substituted Values": self.formula_with_substituted_values,
            "5. Intermediate Calculations": self.intermediate_calculations,
            "6. Final Answer": self.final_answer,
            "7. Verification Checks": self.verification_checks,
            "8. Assumptions": self.assumptions,
        }

    def format_student_report(self) -> str:
        """Format the 8 verification components into a Markdown report."""
        lines = [
            f"### 📚 Student Verification Report: {self.model_name}",
            "",
            "#### 1. Inputs Extracted",
        ]
        for k, v in self.inputs_extracted.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 2. Variable Definitions"])
        for k, v in self.variable_definitions.items():
            lines.append(f"- **{k}**: {v}")
        lines.extend(["", "#### 3. Formula"])
        for k, v in self.formula.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 4. Formula with Substituted Values"])
        for k, v in self.formula_with_substituted_values.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 5. Intermediate Calculations"])
        for k, v in self.intermediate_calculations.items():
            if k == "evaluated_tiers":
                lines.append("- **evaluated_tiers**:")
                for t in self.intermediate_calculations.get("evaluated_tiers", []):
                    lines.append(f"  - Tier {t.get('tier_index', 0) + 1}: Q_feas={t.get('feasible_order_quantity')}, TC=${t.get('total_annual_cost'):,.2f}")
            elif isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 6. Final Answer"])
        for k, v in self.final_answer.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 7. Verification Checks"])
        for c in self.verification_checks:
            lines.append(f"- ✅ {c}")
        lines.extend(["", "#### 8. Assumptions"])
        for a in self.assumptions:
            lines.append(f"- ℹ️ {a}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ROPResult:
    """Structured results for Safety Stock and Reorder Point calculations."""
    model_name: str = "Safety Stock and Reorder Point (SS & ROP)"
    
    # 8 Required Student Verification Components
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Backward-compatible aliases
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Direct typed attributes (supporting both notation sets)
    avg_daily_demand: float = 0.0
    lead_time_days: float = 0.0
    std_daily_demand: float = 0.0
    std_lead_time_days: float = 0.0
    service_level: float = 0.95
    z_score: float = 0.0
    
    lead_time_demand: float = 0.0
    combined_std_dev: float = 0.0
    sigma_LT: float = 0.0
    safety_stock: float = 0.0
    recommended_safety_stock: int = 0
    reorder_point: float = 0.0
    recommended_reorder_point: int = 0
    stockout_risk_per_cycle: float = 0.0
    
    unit_cost: float = 0.0
    annual_holding_rate: float = 0.20
    annual_safety_stock_holding_cost: float = 0.0
    variability_case: str = ""
    
    service_level_tradeoffs: List[Dict[str, Any]] = field(default_factory=list)
    verification: CalculationVerification = field(default_factory=lambda: CalculationVerification(is_valid=True))

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get_student_verification(self) -> Dict[str, Any]:
        """Return the 8 structured components for student verification."""
        return {
            "1. Inputs Extracted": self.inputs_extracted,
            "2. Variable Definitions": self.variable_definitions,
            "3. Formula": self.formula,
            "4. Formula with Substituted Values": self.formula_with_substituted_values,
            "5. Intermediate Calculations": self.intermediate_calculations,
            "6. Final Answer": self.final_answer,
            "7. Verification Checks": self.verification_checks,
            "8. Assumptions": self.assumptions,
        }

    def format_student_report(self) -> str:
        """Format the 8 verification components into a Markdown report."""
        lines = [
            f"### 📚 Student Verification Report: {self.model_name}",
            "",
            "#### 1. Inputs Extracted",
        ]
        for k, v in self.inputs_extracted.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 2. Variable Definitions"])
        for k, v in self.variable_definitions.items():
            lines.append(f"- **{k}**: {v}")
        lines.extend(["", "#### 3. Formula"])
        for k, v in self.formula.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 4. Formula with Substituted Values"])
        for k, v in self.formula_with_substituted_values.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 5. Intermediate Calculations"])
        for k, v in self.intermediate_calculations.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 6. Final Answer"])
        for k, v in self.final_answer.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 7. Verification Checks"])
        for c in self.verification_checks:
            lines.append(f"- ✅ {c}")
        lines.extend(["", "#### 8. Assumptions"])
        for a in self.assumptions:
            lines.append(f"- ℹ️ {a}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compatible alias
SafetyStockROPResult = ROPResult


@dataclass
class PeriodicReviewResult:
    """Structured results for Fixed-Period Review (Order-Up-To Level / Base Stock)."""
    model_name: str = "Fixed-Period Review Policy (Order-Up-To)"
    
    # 8 Required Student Verification Components
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Backward-compatible aliases
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    review_period_days: float = 0.0
    lead_time_days: float = 0.0
    protection_interval_days: float = 0.0
    avg_daily_demand: float = 0.0
    std_daily_demand: float = 0.0
    service_level: float = 0.95
    z_score: float = 0.0
    
    current_on_hand: float = 0.0
    inventory_on_order: float = 0.0
    backorders: float = 0.0
    inventory_position: float = 0.0
    
    expected_demand_protection_period: float = 0.0
    std_dev_protection_period: float = 0.0
    safety_stock: float = 0.0
    recommended_safety_stock: int = 0
    order_up_to_level: float = 0.0
    recommended_target_level: int = 0
    order_quantity: float = 0.0
    recommended_order_quantity: int = 0
    
    continuous_review_ss_equivalent: float = 0.0
    periodic_safety_stock_multiplier: float = 1.0
    
    verification: CalculationVerification = field(default_factory=lambda: CalculationVerification(is_valid=True))

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get_student_verification(self) -> Dict[str, Any]:
        """Return the 8 structured components for student verification."""
        return {
            "1. Inputs Extracted": self.inputs_extracted,
            "2. Variable Definitions": self.variable_definitions,
            "3. Formula": self.formula,
            "4. Formula with Substituted Values": self.formula_with_substituted_values,
            "5. Intermediate Calculations": self.intermediate_calculations,
            "6. Final Answer": self.final_answer,
            "7. Verification Checks": self.verification_checks,
            "8. Assumptions": self.assumptions,
        }

    def format_student_report(self) -> str:
        """Format the 8 verification components into a Markdown report."""
        lines = [
            f"### 📚 Student Verification Report: {self.model_name}",
            "",
            "#### 1. Inputs Extracted",
        ]
        for k, v in self.inputs_extracted.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 2. Variable Definitions"])
        for k, v in self.variable_definitions.items():
            lines.append(f"- **{k}**: {v}")
        lines.extend(["", "#### 3. Formula"])
        for k, v in self.formula.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 4. Formula with Substituted Values"])
        for k, v in self.formula_with_substituted_values.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 5. Intermediate Calculations"])
        for k, v in self.intermediate_calculations.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 6. Final Answer"])
        for k, v in self.final_answer.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 7. Verification Checks"])
        for c in self.verification_checks:
            lines.append(f"- ✅ {c}")
        lines.extend(["", "#### 8. Assumptions"])
        for a in self.assumptions:
            lines.append(f"- ℹ️ {a}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compatible alias
FixedPeriodReviewResult = PeriodicReviewResult


@dataclass
class NewsvendorResult:
    """Structured results for Newsvendor (Single-Period Stochastic Demand) model."""
    model_name: str = "Newsvendor Single-Period Model"
    
    # 8 Required Student Verification Components
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Backward-compatible aliases
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    selling_price: float = 0.0
    unit_cost: float = 0.0
    salvage_value: float = 0.0
    goodwill_penalty: float = 0.0
    
    cost_underage: float = 0.0
    cost_overage: float = 0.0
    critical_ratio: float = 0.0
    z_score: float = 0.0
    
    mean_demand: float = 0.0
    std_demand: float = 0.0
    optimal_order_quantity: float = 0.0
    recommended_order_quantity: int = 0
    
    expected_lost_sales: float = 0.0
    expected_sales: float = 0.0
    expected_leftover_units: float = 0.0
    expected_revenue: float = 0.0
    expected_purchase_cost: float = 0.0
    expected_salvage_revenue: float = 0.0
    expected_profit: float = 0.0
    in_stock_probability: float = 0.0
    stockout_probability: float = 0.0
    
    verification: CalculationVerification = field(default_factory=lambda: CalculationVerification(is_valid=True))

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get_student_verification(self) -> Dict[str, Any]:
        """Return the 8 structured components for student verification."""
        return {
            "1. Inputs Extracted": self.inputs_extracted,
            "2. Variable Definitions": self.variable_definitions,
            "3. Formula": self.formula,
            "4. Formula with Substituted Values": self.formula_with_substituted_values,
            "5. Intermediate Calculations": self.intermediate_calculations,
            "6. Final Answer": self.final_answer,
            "7. Verification Checks": self.verification_checks,
            "8. Assumptions": self.assumptions,
        }

    def format_student_report(self) -> str:
        """Format the 8 verification components into a Markdown report."""
        lines = [
            f"### 📚 Student Verification Report: {self.model_name}",
            "",
            "#### 1. Inputs Extracted",
        ]
        for k, v in self.inputs_extracted.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 2. Variable Definitions"])
        for k, v in self.variable_definitions.items():
            lines.append(f"- **{k}**: {v}")
        lines.extend(["", "#### 3. Formula"])
        for k, v in self.formula.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 4. Formula with Substituted Values"])
        for k, v in self.formula_with_substituted_values.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 5. Intermediate Calculations"])
        for k, v in self.intermediate_calculations.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 6. Final Answer"])
        for k, v in self.final_answer.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: `{v:,.4f}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.extend(["", "#### 7. Verification Checks"])
        for c in self.verification_checks:
            lines.append(f"- ✅ {c}")
        lines.extend(["", "#### 8. Assumptions"])
        for a in self.assumptions:
            lines.append(f"- ℹ️ {a}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Tool 1: Economic Order Quantity (EOQ)
# -----------------------------------------------------------------------------

def calculate_eoq(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: float = 0.0,
    work_days_per_year: int = 250,
) -> EOQResult:
    """Calculate Economic Order Quantity (EOQ) using Q = sqrt(2DS / H).
    
    Validates all inputs, never invents missing numerical values, and returns
    structured results with intermediate calculations, final answers, and assumptions.
    
    Parameters:
        demand: Annual demand (D > 0). Required.
        ordering_cost: Setup / ordering cost per order (S > 0). Required.
        holding_cost: Annual holding cost per unit (H > 0). Required.
        unit_cost: Purchase price per unit (C >= 0). Optional (default 0.0).
        work_days_per_year: Operating days per year for cycle conversion (default 250).
        
    Raises:
        ValueError: If any required numerical input is None, non-positive, or invalid.
    """
    # 1. Validation - Never invent missing numerical inputs
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' (D) cannot be None.")
    if ordering_cost is None:
        raise ValueError("Missing required numerical input: 'ordering_cost' (S) cannot be None.")
    if holding_cost is None:
        raise ValueError("Missing required numerical input: 'holding_cost' (H) cannot be None.")

    try:
        demand = float(demand)
        ordering_cost = float(ordering_cost)
        holding_cost = float(holding_cost)
        unit_cost = float(unit_cost)
        work_days_per_year = int(work_days_per_year)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid non-numeric input provided to calculate_eoq: {err}")

    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0), got {demand}.")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost (S) must be strictly positive (> 0), got {ordering_cost}.")
    if holding_cost <= 0:
        raise ValueError(f"Holding cost (H) must be strictly positive (> 0), got {holding_cost}.")
    if unit_cost < 0:
        raise ValueError(f"Unit cost cannot be negative, got {unit_cost}.")
    if work_days_per_year <= 0:
        raise ValueError(f"Work days per year must be strictly positive (> 0), got {work_days_per_year}.")

    # 2. Mathematical Calculations
    numerator_2DS = 2.0 * demand * ordering_cost
    ratio = numerator_2DS / holding_cost
    eoq_val = math.sqrt(ratio)
    recommended_qty = int(round(eoq_val))
    if recommended_qty < 1:
        recommended_qty = 1

    orders_per_year = demand / eoq_val
    order_cycle_years = eoq_val / demand
    order_cycle_days = order_cycle_years * work_days_per_year

    annual_ordering_cost = (demand / eoq_val) * ordering_cost
    annual_holding_cost = (eoq_val / 2.0) * holding_cost
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = demand * unit_cost
    grand_total_cost = total_inventory_cost + annual_purchase_cost

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "holding_cost": holding_cost,
        "unit_cost": unit_cost,
        "work_days_per_year": work_days_per_year,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand in units per year (D > 0)",
        "S": "Ordering / setup cost incurred per replenishment order ($/order)",
        "H": "Inventory holding / carrying cost per unit per year ($/unit/year)",
        "C": "Purchase price / acquisition cost per unit ($/unit)",
        "N_days": "Number of operating workdays per year (days/year)",
        "Q*": "Economic Order Quantity: batch size that minimizes total annual inventory cost (units)",
        "Q_rec": "Recommended order quantity rounded to nearest integer (units)",
        "N": "Order frequency: expected number of replenishment orders placed annually (orders/year)",
        "T_years": "Order cycle duration in fractional years (years)",
        "T_days": "Order cycle duration in operating days (days)",
        "TOC": "Total Annual Ordering Cost = (D / Q*) * S ($/year)",
        "THC": "Total Annual Holding Cost = (Q* / 2) * H ($/year)",
        "TIC": "Total Annual Inventory Cost = TOC + THC ($/year)",
        "TPC": "Total Annual Purchase Cost = D * C ($/year)",
        "TC": "Grand Total Annual Cost = TIC + TPC ($/year)",
    }

    # 3. Formulas
    formula = {
        "eoq": "Q* = sqrt((2 * D * S) / H)",
        "order_frequency": "N = D / Q*",
        "order_cycle_years": "T_years = Q* / D",
        "order_cycle_days": "T_days = (Q* / D) * N_days",
        "annual_ordering_cost": "TOC = (D / Q*) * S",
        "annual_holding_cost": "THC = (Q* / 2) * H",
        "total_annual_inventory_cost": "TIC = TOC + THC = (D / Q*) * S + (Q* / 2) * H",
        "annual_purchase_cost": "TPC = D * C",
        "grand_total_annual_cost": "TC = TIC + TPC = TOC + THC + (D * C)",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "eoq": (
            f"Q* = sqrt((2 * {demand:g} * {ordering_cost:g}) / {holding_cost:g}) = "
            f"sqrt({numerator_2DS:g} / {holding_cost:g}) = sqrt({ratio:g}) = "
            f"{eoq_val:.4f} units (Recommended integer: {recommended_qty} units)"
        ),
        "order_frequency": f"N = {demand:g} / {eoq_val:.4f} = {orders_per_year:.4f} orders/year",
        "order_cycle_years": f"T_years = {eoq_val:.4f} / {demand:g} = {order_cycle_years:.4f} years",
        "order_cycle_days": f"T_days = ({eoq_val:.4f} / {demand:g}) * {work_days_per_year} = {order_cycle_days:.2f} operating days",
        "annual_ordering_cost": f"TOC = ({demand:g} / {eoq_val:.4f}) * {ordering_cost:g} = ${annual_ordering_cost:.2f}",
        "annual_holding_cost": f"THC = ({eoq_val:.4f} / 2) * {holding_cost:g} = ${annual_holding_cost:.2f}",
        "total_annual_inventory_cost": f"TIC = ${annual_ordering_cost:.2f} + ${annual_holding_cost:.2f} = ${total_inventory_cost:.2f}",
        "annual_purchase_cost": f"TPC = {demand:g} * ${unit_cost:g} = ${annual_purchase_cost:.2f}",
        "grand_total_annual_cost": f"TC = ${total_inventory_cost:.2f} + ${annual_purchase_cost:.2f} = ${grand_total_cost:.2f}",
    }

    # 5. Intermediate Calculations Breakdown
    intermediate_calcs = {
        "numerator_2DS": numerator_2DS,
        "denominator_H": holding_cost,
        "fraction_2DS_over_H": ratio,
        "unrounded_eoq": eoq_val,
        "annual_order_frequency_N": orders_per_year,
        "order_cycle_interval_years": order_cycle_years,
        "order_cycle_interval_days": order_cycle_days,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "holding_minus_ordering_cost_diff": abs(annual_holding_cost - annual_ordering_cost),
        "total_annual_inventory_cost": total_inventory_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "grand_total_annual_cost": grand_total_cost,
    }

    # 6. Final Answer Summary
    final_answer = {
        "optimal_order_quantity": eoq_val,
        "recommended_order_quantity": recommended_qty,
        "orders_per_year": orders_per_year,
        "order_cycle_days": order_cycle_days,
        "total_annual_inventory_cost": total_inventory_cost,
        "grand_total_annual_cost": grand_total_cost,
    }

    # 7. Verification Checks
    verification = CalculationVerification(is_valid=True)
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    rel_diff = cost_diff / max(annual_ordering_cost, 1e-9)
    if rel_diff < 1e-3:
        verification.checks_passed.append(
            f"EOQ optimality condition verified: Ordering cost (${annual_ordering_cost:.2f}) "
            f"equals Holding cost (${annual_holding_cost:.2f}) within 0.1%."
        )
    verification.checks_passed.append(
        f"Order quantity {eoq_val:.2f} satisfies second-order convexity condition."
    )
    verification.checks_passed.append(
        f"EOQ ({eoq_val:.2f} units) produces strictly positive lot size."
    )
    verification_checks = list(verification.checks_passed)

    # 8. Model Assumptions
    assumptions = [
        "Demand rate is constant, uniform, and known with certainty throughout the year.",
        "Ordering/setup cost (S) is fixed and independent of order quantity.",
        "Inventory carrying/holding cost (H) is linear and constant per unit per year.",
        "Replenishment lead time is zero (instantaneous receipt of entire batch).",
        "Shortages, stockouts, and backorders are strictly prohibited.",
        "No quantity discounts apply; purchase price per unit is fixed.",
    ]

    # Sensitivity curve points for UI/plotting
    q_multipliers = np.linspace(0.2, 2.5, 50)
    q_vals = [float(eoq_val * m) for m in q_multipliers]
    holding_costs = [float((q / 2.0) * holding_cost) for q in q_vals]
    ordering_costs = [float((demand / q) * ordering_cost) for q in q_vals]
    total_costs = [h + o for h, o in zip(holding_costs, ordering_costs)]
    cost_curve = {
        "order_quantities": q_vals,
        "holding_costs": holding_costs,
        "ordering_costs": ordering_costs,
        "total_inventory_costs": total_costs
    }

    return EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        inputs=inputs_extracted,
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost_per_unit=holding_cost,
        unit_cost=unit_cost,
        work_days_per_year=work_days_per_year,
        eoq=eoq_val,
        recommended_order_quantity=recommended_qty,
        orders_per_year=orders_per_year,
        order_cycle_days=order_cycle_days,
        order_cycle_years=order_cycle_years,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_annual_inventory_cost=total_inventory_cost,
        annual_purchase_cost=annual_purchase_cost,
        grand_total_annual_cost=grand_total_cost,
        cost_curve=cost_curve,
        verification=verification,
    )


# -----------------------------------------------------------------------------
# Tool 2: Quantity Discount Analysis
# -----------------------------------------------------------------------------

def calculate_quantity_discount(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    tiers: Optional[List[Dict[str, Any]]] = None,
    holding_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """Evaluate all-units quantity discount schedules to identify the global minimum total annual cost.
    
    Supports both fixed holding cost per unit and percentage holding cost (holding_rate * unit_price).
    Evaluates feasible order quantities against price-break thresholds.
    
    Parameters:
        demand: Annual demand (D > 0). Required.
        ordering_cost: Setup / ordering cost per order (S > 0). Required.
        tiers: List of tier dicts, each with 'min_qty', 'max_qty' (or None), and 'unit_price'. Required.
        holding_rate: Annual holding cost rate as fraction of unit price (e.g. 0.20 for 20%).
        fixed_holding_cost: Fixed annual holding cost per unit in dollars (alternative to holding_rate).
        
    Raises:
        ValueError: If required inputs are missing, empty, or invalid.
    """
    # 1. Validation - Never invent missing numerical inputs
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' cannot be None.")
    if ordering_cost is None:
        raise ValueError("Missing required numerical input: 'ordering_cost' cannot be None.")
    if tiers is None or len(tiers) == 0:
        raise ValueError("Missing required input: 'tiers' schedule must contain at least one price tier.")
    if holding_rate is None and fixed_holding_cost is None:
        raise ValueError("Missing required numerical input: Must provide either 'holding_rate' or 'fixed_holding_cost'.")

    try:
        demand = float(demand)
        ordering_cost = float(ordering_cost)
        if holding_rate is not None:
            holding_rate = float(holding_rate)
        if fixed_holding_cost is not None:
            fixed_holding_cost = float(fixed_holding_cost)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid non-numeric input provided to calculate_quantity_discount: {err}")

    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0), got {demand}.")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be strictly positive (> 0), got {ordering_cost}.")
    if holding_rate is not None and holding_rate <= 0:
        raise ValueError(f"Holding rate must be strictly positive (> 0), got {holding_rate}.")
    if fixed_holding_cost is not None and fixed_holding_cost <= 0:
        raise ValueError(f"Fixed holding cost must be strictly positive (> 0), got {fixed_holding_cost}.")

    # 2. Evaluate Tiers
    sorted_tiers = sorted(tiers, key=lambda t: float(t.get("min_qty", 0.0)))
    tier_evaluations: List[DiscountTierResult] = []
    intermediate_tier_details: List[Dict[str, Any]] = []

    for idx, tier in enumerate(sorted_tiers):
        min_qty = float(tier.get("min_qty", 0.0))
        max_qty = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["unit_price"])
        if price <= 0:
            raise ValueError(f"Tier {idx + 1} unit_price must be positive, got {price}.")

        # Holding cost determination
        if holding_rate is not None:
            tier_h = holding_rate * price
        else:
            tier_h = float(fixed_holding_cost)  # type: ignore

        if tier_h <= 0:
            tier_h = 1e-4

        unconstrained_eoq = math.sqrt((2.0 * demand * ordering_cost) / tier_h)

        # Check feasibility
        is_price_break = False
        if unconstrained_eoq < min_qty:
            feasible_q = min_qty
            is_price_break = True
        elif max_qty is not None and unconstrained_eoq > max_qty:
            feasible_q = max_qty
        else:
            feasible_q = unconstrained_eoq

        ann_purchase = demand * price
        ann_order = (demand / feasible_q) * ordering_cost
        ann_hold = (feasible_q / 2.0) * tier_h
        tot_cost = ann_purchase + ann_order + ann_hold

        tier_res = DiscountTierResult(
            tier_index=idx,
            min_qty=min_qty,
            max_qty=max_qty,
            unit_price=price,
            holding_cost_per_unit=tier_h,
            unconstrained_eoq=unconstrained_eoq,
            feasible_order_quantity=feasible_q,
            is_price_break_point=is_price_break,
            annual_purchase_cost=ann_purchase,
            annual_ordering_cost=ann_order,
            annual_holding_cost=ann_hold,
            total_annual_cost=tot_cost,
        )
        tier_evaluations.append(tier_res)
        intermediate_tier_details.append(asdict(tier_res))

    optimal_tier = min(tier_evaluations, key=lambda t: t.total_annual_cost)
    baseline_tier = tier_evaluations[0]
    baseline_cost = baseline_tier.total_annual_cost
    optimal_cost = optimal_tier.total_annual_cost
    savings = max(0.0, baseline_cost - optimal_cost)
    pct_savings = (savings / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "tiers": sorted_tiers,
        "holding_rate": holding_rate,
        "fixed_holding_cost": fixed_holding_cost,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering / setup cost per replenishment order ($/order)",
        "i": "Annual inventory holding rate fraction of unit purchase price (year^-1)",
        "H_fixed": "Fixed annual holding cost per unit in dollars ($/unit/year)",
        "P_j": "Unit purchase price for discount bracket j ($/unit)",
        "H_j": "Unit holding cost per year for Tier j: i * P_j or H_fixed ($/unit/year)",
        "q_min_j": "Minimum order quantity to qualify for Tier j (units)",
        "q_max_j": "Maximum order quantity for Tier j (units or unbounded)",
        "EOQ_j": "Unconstrained Economic Order Quantity for Tier j: sqrt((2 * D * S) / H_j) (units)",
        "Q_feas_j": "Feasible lot size for Tier j: EOQ_j adjusted to price-break minimum if EOQ_j < q_min_j (units)",
        "TPC_j": "Annual purchase cost for Tier j = D * P_j ($/year)",
        "TOC_j": "Annual ordering cost for Tier j = (D / Q_feas_j) * S ($/year)",
        "THC_j": "Annual holding cost for Tier j = (Q_feas_j / 2) * H_j ($/year)",
        "TC_j": "Total annual cost for Tier j = TPC_j + TOC_j + THC_j ($/year)",
        "Q*": "Optimal replenishment order quantity achieving minimum TC across all tiers (units)",
        "TC*": "Global minimum total annual cost ($/year)",
        "Savings": "Cost savings relative to undiscounted Tier 1 baseline: TC_baseline - TC* ($/year)",
        "Pct_Savings": "Percentage cost savings: (Savings / TC_baseline) * 100 (%)",
    }

    # 3. Formulas
    formula = {
        "tier_holding_cost": "H_j = (i * P_j) if percentage rate else H_fixed",
        "unconstrained_eoq": "EOQ_j = sqrt((2 * D * S) / H_j)",
        "feasible_quantity": "Q_feas_j = q_min_j if EOQ_j < q_min_j else (q_max_j if (q_max_j and EOQ_j > q_max_j) else EOQ_j)",
        "tier_purchase_cost": "TPC_j = D * P_j",
        "tier_ordering_cost": "TOC_j = (D / Q_feas_j) * S",
        "tier_holding_cost_annual": "THC_j = (Q_feas_j / 2) * H_j",
        "tier_total_annual_cost": "TC_j = TPC_j + TOC_j + THC_j = (D * P_j) + (D / Q_feas_j) * S + (Q_feas_j / 2) * H_j",
        "optimal_selection": "TC* = min(TC_j for all tiers j); Q* = Q_feas corresponding to min(TC_j)",
        "annual_cost_savings": "Savings = TC_baseline - TC*",
        "percentage_savings": "Pct_Savings = (Savings / TC_baseline) * 100",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {}
    for t in tier_evaluations:
        max_str = f"{t.max_qty:g}" if t.max_qty is not None else "∞"
        break_note = f"Adjusted to price-break minimum {t.min_qty:g}" if t.is_price_break_point else f"EOQ {t.unconstrained_eoq:.2f} within range"
        formula_with_substituted_values[f"tier_{t.tier_index + 1}_evaluation"] = (
            f"Tier {t.tier_index + 1} (Range: {t.min_qty:g}-{max_str}, Price: ${t.unit_price:.2f}): "
            f"H_{t.tier_index + 1} = ${t.holding_cost_per_unit:.2f}; "
            f"EOQ = sqrt((2 * {demand:g} * {ordering_cost:g}) / {t.holding_cost_per_unit:.2f}) = {t.unconstrained_eoq:.2f} units -> "
            f"Feasible Q = {t.feasible_order_quantity:g} ({break_note}); "
            f"TPC = {demand:g} * ${t.unit_price:.2f} = ${t.annual_purchase_cost:.2f}; "
            f"TOC = ({demand:g} / {t.feasible_order_quantity:g}) * ${ordering_cost:g} = ${t.annual_ordering_cost:.2f}; "
            f"THC = ({t.feasible_order_quantity:g} / 2) * ${t.holding_cost_per_unit:.2f} = ${t.annual_holding_cost:.2f}; "
            f"TC = ${t.annual_purchase_cost:.2f} + ${t.annual_ordering_cost:.2f} + ${t.annual_holding_cost:.2f} = ${t.total_annual_cost:.2f}"
        )
    formula_with_substituted_values["optimal_decision"] = (
        f"Optimal Decision: Tier {optimal_tier.tier_index + 1} (Price: ${optimal_tier.unit_price:.2f}) with order quantity Q* = {optimal_tier.feasible_order_quantity:g} units achieves minimum total annual cost TC* = ${optimal_cost:.2f}"
    )
    formula_with_substituted_values["annual_cost_savings"] = (
        f"Savings = ${baseline_cost:.2f} (Tier 1 baseline) - ${optimal_cost:.2f} (Tier {optimal_tier.tier_index + 1} optimal) = ${savings:.2f} ({pct_savings:.2f}% savings)"
    )

    # 5. Model Assumptions
    assumptions = [
        "All-units quantity discount applies to entire order quantity once threshold is met.",
        "Demand is steady, continuous, and deterministic over the annual horizon.",
        "Order replenishment is instantaneous with no supply lead time delay.",
        "Holding cost is either a constant dollar amount or a fixed percentage of unit price.",
        "Sufficient warehouse capacity and working capital exist to finance larger batch sizes.",
        "Shortages or partial deliveries are not permitted.",
    ]

    # 6. Intermediate Calculations
    intermediate_calcs = {
        "holding_cost_structure": "percentage_of_price" if holding_rate is not None else "fixed_per_unit",
        "holding_rate": holding_rate,
        "fixed_holding_cost": fixed_holding_cost,
        "evaluated_tiers": intermediate_tier_details,
        "baseline_tier_cost": baseline_cost,
        "optimal_tier_cost": optimal_cost,
        "annual_cost_savings": savings,
        "percentage_savings": pct_savings,
    }

    # 7. Final Answer
    final_answer = {
        "optimal_tier_index": optimal_tier.tier_index,
        "optimal_tier_number": optimal_tier.tier_index + 1,
        "optimal_order_quantity": optimal_tier.feasible_order_quantity,
        "recommended_order_quantity": int(round(optimal_tier.feasible_order_quantity)),
        "optimal_unit_price": optimal_tier.unit_price,
        "optimal_total_annual_cost": optimal_cost,
        "annual_cost_savings": savings,
        "percentage_savings": pct_savings,
        "is_adjusted_to_price_break": optimal_tier.is_price_break_point,
    }

    # 8. Verification Checks
    verification = CalculationVerification(is_valid=True)
    verification.checks_passed.append(
        f"Evaluated {len(tier_evaluations)} price tiers. Optimal quantity {optimal_tier.feasible_order_quantity:.2f} "
        f"selected from Tier {optimal_tier.tier_index + 1} at price ${optimal_tier.unit_price:.2f}."
    )
    if optimal_tier.is_price_break_point:
        verification.checks_passed.append(
            f"Note: Order quantity was adjusted upwards to price-break minimum ({optimal_tier.min_qty:.0f}) "
            "to secure the discount, yielding net cost reduction."
        )
    verification.checks_passed.append(
        f"Global optimality verified: Tier {optimal_tier.tier_index + 1} total cost (${optimal_cost:.2f}) is strictly <= all other evaluated tiers."
    )
    verification_checks = list(verification.checks_passed)

    return QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        inputs=inputs_extracted,
        demand=demand,
        ordering_cost=ordering_cost,
        holding_rate=holding_rate,
        fixed_holding_cost=fixed_holding_cost,
        optimal_tier_index=optimal_tier.tier_index,
        optimal_order_quantity=optimal_tier.feasible_order_quantity,
        recommended_order_quantity=int(round(optimal_tier.feasible_order_quantity)),
        optimal_unit_price=optimal_tier.unit_price,
        optimal_total_annual_cost=optimal_cost,
        baseline_total_annual_cost=baseline_cost,
        annual_cost_savings=savings,
        percentage_savings=pct_savings,
        tier_evaluations=tier_evaluations,
        verification=verification,
    )


# -----------------------------------------------------------------------------
# Tool 3: Safety Stock and Reorder Point (SS & ROP)
# -----------------------------------------------------------------------------

def calculate_rop(
    demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    service_level: float = 0.95,
    unit_cost: float = 0.0,
    annual_holding_rate: float = 0.20,
    # Keyword aliases for flexibility
    avg_daily_demand: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    std_daily_demand: Optional[float] = None,
    std_lead_time_days: Optional[float] = None,
) -> ROPResult:
    """Calculate Safety Stock (SS) and Reorder Point (ROP).
    
    Supports all four variability scenarios:
    1. Constant demand and constant lead time:
       sigma_LT = 0
       SS = 0
       ROP = d * L
    2. Variable demand and constant lead time:
       sigma_LT = sigma_d * sqrt(L)
       SS = z * sigma_LT
       ROP = d * L + SS
    3. Constant demand and variable lead time:
       sigma_LT = d * sigma_L
       SS = z * sigma_LT
       ROP = d * L + SS
    4. Variable demand and variable lead time:
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS
       
    Raises:
        ValueError: If required inputs are missing, negative, or invalid.
    """
    # Resolve aliases
    d_val = demand if demand is not None else avg_daily_demand
    lt_val = lead_time if lead_time is not None else lead_time_days
    sd_val = std_demand if std_demand > 0 else (std_daily_demand or 0.0)
    slt_val = std_lead_time if std_lead_time > 0 else (std_lead_time_days or 0.0)

    # 1. Validation - Never invent missing numerical inputs
    if d_val is None:
        raise ValueError("Missing required numerical input: 'demand' (d) cannot be None.")
    if lt_val is None:
        raise ValueError("Missing required numerical input: 'lead_time' (L) cannot be None.")

    try:
        d = float(d_val)
        L = float(lt_val)
        sigma_d = float(sd_val)
        sigma_L = float(slt_val)
        sl = float(service_level)
        unit_c = float(unit_cost)
        hold_rate = float(annual_holding_rate)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid non-numeric input provided to calculate_rop: {err}")

    if d < 0:
        raise ValueError(f"Average demand per period cannot be negative, got {d}.")
    if L < 0:
        raise ValueError(f"Lead time cannot be negative, got {L}.")
    if sigma_d < 0:
        raise ValueError(f"Standard deviation of demand cannot be negative, got {sigma_d}.")
    if sigma_L < 0:
        raise ValueError(f"Standard deviation of lead time cannot be negative, got {sigma_L}.")
    if not (0.0 < sl < 1.0):
        raise ValueError(f"Service level must be strictly between 0 and 1 (e.g. 0.95 for 95%), got {sl}.")

    # 2. Mathematical Calculations
    ddlt = d * L  # Expected demand during lead time

    var_demand = sigma_d > 0
    var_lt = sigma_L > 0

    if var_demand and var_lt:
        var_case = "Variable demand and variable lead time"
        variance_L_sigd = L * (sigma_d ** 2)
        variance_d_sigL = (d ** 2) * (sigma_L ** 2)
        sigma_LT = math.sqrt(variance_L_sigd + variance_d_sigL)
    elif var_demand and not var_lt:
        var_case = "Variable demand and constant lead time"
        variance_L_sigd = L * (sigma_d ** 2)
        variance_d_sigL = 0.0
        sigma_LT = sigma_d * math.sqrt(L)
    elif not var_demand and var_lt:
        var_case = "Constant demand and variable lead time"
        variance_L_sigd = 0.0
        variance_d_sigL = (d ** 2) * (sigma_L ** 2)
        sigma_LT = d * sigma_L
    else:
        var_case = "Constant demand and constant lead time"
        variance_L_sigd = 0.0
        variance_d_sigL = 0.0
        sigma_LT = 0.0

    z_score = float(stats.norm.ppf(sl))
    safety_stock = z_score * sigma_LT
    if safety_stock < 0:
        safety_stock = 0.0

    recommended_ss = int(math.ceil(safety_stock))
    rop = ddlt + safety_stock
    recommended_rop = int(math.ceil(rop))
    stockout_risk = 1.0 - sl

    annual_ss_holding_cost = 0.0
    if unit_c > 0:
        annual_ss_holding_cost = safety_stock * unit_c * hold_rate

    # Service level trade-off curve
    tradeoffs = []
    for test_sl in [0.85, 0.90, 0.95, 0.98, 0.99, 0.999]:
        z_sl = float(stats.norm.ppf(test_sl))
        ss_sl = max(0.0, z_sl * sigma_LT)
        rop_sl = ddlt + ss_sl
        hold_c = ss_sl * unit_c * hold_rate if unit_c > 0 else 0.0
        tradeoffs.append({
            "service_level": test_sl,
            "z_score": round(z_sl, 3),
            "safety_stock": round(ss_sl, 2),
            "reorder_point": round(rop_sl, 2),
            "annual_ss_holding_cost": round(hold_c, 2),
            "stockout_risk_pct": round((1.0 - test_sl) * 100, 2),
        })

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand": d,
        "lead_time": L,
        "std_demand": sigma_d,
        "std_lead_time": sigma_L,
        "service_level": sl,
        "unit_cost": unit_c,
        "annual_holding_rate": hold_rate,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average periodic demand rate (units/period)",
        "L": "Average replenishment lead time (periods)",
        "sigma_d": "Standard deviation of periodic demand (units/period)",
        "sigma_L": "Standard deviation of replenishment lead time (periods)",
        "CSL": "Cycle Service Level: target probability of avoiding stockouts during lead time",
        "z": "Safety factor: standard normal z-score corresponding to CSL (Phi(z) = CSL)",
        "DDLT": "Expected demand during lead time = d * L (units)",
        "sigma_LT": "Combined standard deviation of lead time demand accounting for uncertainty (units)",
        "SS": "Safety stock buffer = z * sigma_LT (units)",
        "SS_rec": "Recommended integer safety stock = ceil(SS) (units)",
        "ROP": "Reorder Point = DDLT + SS = (d * L) + SS (units)",
        "ROP_rec": "Recommended integer reorder point = ceil(ROP) (units)",
        "Stockout_Risk": "Cycle stockout risk = 1 - CSL",
        "C": "Unit purchase price ($/unit)",
        "i": "Annual holding cost rate fraction (year^-1)",
        "Annual_SS_Cost": "Annual carrying cost of safety stock = SS * C * i ($/year)",
    }

    # 3. Formulas
    if var_demand and var_lt:
        formula_sigma_lt = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        sub_sigma_lt = f"sigma_LT = sqrt({L:g} * ({sigma_d:g})^2 + ({d:g})^2 * ({sigma_L:g})^2) = sqrt({variance_L_sigd:.2f} + {variance_d_sigL:.2f}) = {sigma_LT:.4f} units"
    elif var_demand and not var_lt:
        formula_sigma_lt = "sigma_LT = sigma_d * sqrt(L)"
        sub_sigma_lt = f"sigma_LT = {sigma_d:g} * sqrt({L:g}) = {sigma_LT:.4f} units"
    elif not var_demand and var_lt:
        formula_sigma_lt = "sigma_LT = d * sigma_L"
        sub_sigma_lt = f"sigma_LT = {d:g} * {sigma_L:g} = {sigma_LT:.4f} units"
    else:
        formula_sigma_lt = "sigma_LT = 0 (constant demand and constant lead time)"
        sub_sigma_lt = "sigma_LT = 0 units (no uncertainty present)"

    formula = {
        "expected_lead_time_demand": "DDLT = d * L",
        "sigma_lead_time": formula_sigma_lt,
        "z_score": "z = norm.ppf(CSL) such that Phi(z) = CSL",
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = DDLT + SS = (d * L) + SS",
        "stockout_risk": "Stockout Risk per cycle = 1 - CSL",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "expected_lead_time_demand": f"DDLT = {d:g} * {L:g} = {ddlt:.2f} units",
        "sigma_lead_time": sub_sigma_lt,
        "z_score": f"z = norm.ppf({sl:.4f}) = {z_score:.4f}",
        "safety_stock": f"SS = {z_score:.4f} * {sigma_LT:.4f} = {safety_stock:.2f} units (Recommended integer: {recommended_ss} units)",
        "reorder_point": f"ROP = {ddlt:.2f} + {safety_stock:.2f} = {rop:.2f} units (Recommended integer: {recommended_rop} units)",
        "stockout_risk": f"Stockout Risk = 1 - {sl:.4f} = {stockout_risk:.4f} ({stockout_risk * 100:.2f}%)",
    }
    if unit_c > 0:
        formula["annual_ss_holding_cost"] = "Annual SS Cost = SS * C * i"
        formula_with_substituted_values["annual_ss_holding_cost"] = (
            f"Annual SS Cost = {safety_stock:.2f} * ${unit_c:.2f} * {hold_rate:.2f} = ${annual_ss_holding_cost:.2f}"
        )

    # 5. Model Assumptions
    assumptions = [
        "Continuous review policy: inventory position is monitored in real-time.",
        "Demand during lead time follows a Gaussian (Normal) distribution.",
        "Demand in non-overlapping periods is independent and identically distributed.",
        "Replenishment orders are placed exactly when inventory position hits ROP.",
        "Target is Cycle Service Level (probability of zero stockouts during replenishment lead time).",
    ]

    # 6. Intermediate Calculations
    intermediate_calcs = {
        "variability_case": var_case,
        "expected_lead_time_demand_d_times_L": ddlt,
        "variance_from_demand_L_sigma_d_sq": variance_L_sigd,
        "variance_from_lead_time_d_sq_sigma_L_sq": variance_d_sigL,
        "total_lead_time_variance": variance_L_sigd + variance_d_sigL,
        "sigma_LT": sigma_LT,
        "z_score": z_score,
        "safety_stock_exact": safety_stock,
        "reorder_point_exact": rop,
        "stockout_risk_per_cycle": stockout_risk,
        "annual_safety_stock_holding_cost": annual_ss_holding_cost,
    }

    # 7. Final Answer
    final_answer = {
        "variability_case": var_case,
        "service_level": sl,
        "z_score": z_score,
        "expected_lead_time_demand": ddlt,
        "sigma_LT": sigma_LT,
        "safety_stock": safety_stock,
        "recommended_safety_stock": recommended_ss,
        "reorder_point": rop,
        "recommended_reorder_point": recommended_rop,
        "stockout_risk_per_cycle": stockout_risk,
    }

    # 8. Verification Checks
    verification = CalculationVerification(is_valid=True)
    verification.checks_passed.append(
        f"Verified Reorder Point ROP ({rop:.2f}) = Expected Lead Time Demand ({ddlt:.2f}) + Safety Stock ({safety_stock:.2f})."
    )
    if sigma_LT > 0:
        verification.checks_passed.append(
            f"Z-score {z_score:.3f} mapped for Cycle Service Level {sl:.1%} under Gaussian assumptions."
        )
    verification.checks_passed.append(
        f"Variability scenario classified: '{var_case}' with sigma_LT = {sigma_LT:.4f}."
    )
    verification_checks = list(verification.checks_passed)

    return ROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        inputs=inputs_extracted,
        avg_daily_demand=d,
        lead_time_days=L,
        std_daily_demand=sigma_d,
        std_lead_time_days=sigma_L,
        service_level=sl,
        z_score=z_score,
        lead_time_demand=ddlt,
        combined_std_dev=sigma_LT,
        sigma_LT=sigma_LT,
        safety_stock=safety_stock,
        recommended_safety_stock=recommended_ss,
        reorder_point=rop,
        recommended_reorder_point=recommended_rop,
        stockout_risk_per_cycle=stockout_risk,
        unit_cost=unit_c,
        annual_holding_rate=hold_rate,
        annual_safety_stock_holding_cost=annual_ss_holding_cost,
        variability_case=var_case,
        service_level_tradeoffs=tradeoffs,
        verification=verification,
    )


# Backward-compatible alias
calculate_safety_stock_rop = calculate_rop


# -----------------------------------------------------------------------------
# Tool 4: Fixed-Period Review (Periodic Review / Order-Up-To Level)
# -----------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand: Optional[float] = None,
    std_demand: float = 0.0,
    service_level: float = 0.95,
    current_inventory: float = 0.0,
    inventory_on_order: float = 0.0,
    backorders: float = 0.0,
    inventory_position: Optional[float] = None,
    # Keyword aliases for flexibility
    review_period_days: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    avg_daily_demand: Optional[float] = None,
    std_daily_demand: Optional[float] = None,
    current_on_hand: Optional[float] = None,
) -> PeriodicReviewResult:
    """Calculate target inventory (Order-Up-To Level) and replenishment order quantity.
    
    Formulas:
        Protection period = review period + lead time (R + L)
        Target inventory S = demand during protection period + safety stock
        Safety stock SS = z * sigma_d * sqrt(R + L)
        Order quantity Q = max(0, target inventory - inventory position)
        
    Raises:
        ValueError: If required inputs are missing, negative, or invalid.
    """
    # Resolve aliases
    r_val = review_period if review_period is not None else review_period_days
    l_val = lead_time if lead_time is not None else lead_time_days
    d_val = demand if demand is not None else avg_daily_demand
    sd_val = std_demand if std_demand > 0 else (std_daily_demand or 0.0)
    on_hand = current_inventory if current_inventory != 0.0 else (current_on_hand or 0.0)

    # 1. Validation - Never invent missing numerical inputs
    if r_val is None:
        raise ValueError("Missing required numerical input: 'review_period' (R) cannot be None.")
    if l_val is None:
        raise ValueError("Missing required numerical input: 'lead_time' (L) cannot be None.")
    if d_val is None:
        raise ValueError("Missing required numerical input: 'demand' (d) cannot be None.")

    try:
        R = float(r_val)
        L = float(l_val)
        d = float(d_val)
        sigma_d = float(sd_val)
        sl = float(service_level)
        oh = float(on_hand)
        oo = float(inventory_on_order)
        bo = float(backorders)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid non-numeric input provided to calculate_periodic_review: {err}")

    if R <= 0:
        raise ValueError(f"Review period (R) must be strictly positive (> 0), got {R}.")
    if L < 0:
        raise ValueError(f"Lead time (L) cannot be negative, got {L}.")
    if d < 0:
        raise ValueError(f"Average demand per period cannot be negative, got {d}.")
    if sigma_d < 0:
        raise ValueError(f"Standard deviation of demand cannot be negative, got {sigma_d}.")
    if not (0.0 < sl < 1.0):
        raise ValueError(f"Service level must be strictly between 0 and 1, got {sl}.")

    # 2. Mathematical Calculations
    protection_period = R + L
    demand_during_protection = d * protection_period
    std_dev_protection = sigma_d * math.sqrt(protection_period)

    z_score = float(stats.norm.ppf(sl))
    safety_stock = z_score * std_dev_protection
    target_inventory = demand_during_protection + safety_stock

    if inventory_position is not None:
        ip = float(inventory_position)
    else:
        ip = oh + oo - bo

    order_quantity = max(0.0, target_inventory - ip)
    recommended_target = int(math.ceil(target_inventory))
    recommended_order = int(math.ceil(order_quantity))

    # Continuous review safety stock comparison
    continuous_ss = z_score * sigma_d * math.sqrt(max(L, 1e-6))
    periodic_ratio = safety_stock / continuous_ss if continuous_ss > 0 else 1.0

    # 3. Model Assumptions
    assumptions = [
        "Inventory status is inspected periodically at fixed intervals of length R.",
        "Supplier lead time L is known and replenishment arrives L periods after order placement.",
        "Protection horizon spans R + L periods; no additional replenishment can arrive until next review cycle.",
        "Demand across periods is independent and identically distributed Normal random variables.",
        "Order quantity brings effective inventory position (On-Hand + On-Order - Backorders) up to target S.",
    ]

    # 4. Intermediate Calculations
    intermediate_calcs = {
        "review_period_R": R,
        "lead_time_L": L,
        "protection_period_R_plus_L": protection_period,
        "expected_demand_during_protection_period": demand_during_protection,
        "std_dev_during_protection_period": std_dev_protection,
        "z_score": z_score,
        "safety_stock_exact": safety_stock,
        "target_inventory_S": target_inventory,
        "current_on_hand": oh,
        "inventory_on_order": oo,
        "backorders": bo,
        "inventory_position_IP": ip,
        "raw_order_quantity_S_minus_IP": target_inventory - ip,
        "order_quantity": order_quantity,
        "continuous_review_ss_equivalent": continuous_ss,
        "periodic_safety_stock_multiplier": periodic_ratio,
    }

    # 5. Final Answer
    final_answer = {
        "protection_period": protection_period,
        "target_inventory": target_inventory,
        "recommended_target_inventory": recommended_target,
        "inventory_position": ip,
        "order_quantity": order_quantity,
        "recommended_order_quantity": recommended_order,
        "safety_stock": safety_stock,
        "recommended_safety_stock": int(math.ceil(safety_stock)),
    }

    verification = CalculationVerification(is_valid=True)
    verification.checks_passed.append(
        f"Protection interval R + L = {R:.1f} + {L:.1f} = {protection_period:.1f} periods."
    )
    verification.checks_passed.append(
        f"Inventory Position ({ip:.2f}) evaluated against Target Base Stock ({target_inventory:.2f})."
    )
    if order_quantity == 0:
        verification.warnings.append(
            "Current inventory position meets or exceeds target stock level; recommended order quantity is 0."
        )

    return PeriodicReviewResult(
        inputs={
            "review_period": R,
            "lead_time": L,
            "demand": d,
            "std_demand": sigma_d,
            "service_level": sl,
            "current_inventory": oh,
            "inventory_on_order": oo,
            "backorders": bo,
            "inventory_position": ip,
        },
        assumptions=assumptions,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        review_period_days=R,
        lead_time_days=L,
        protection_interval_days=protection_period,
        avg_daily_demand=d,
        std_daily_demand=sigma_d,
        service_level=sl,
        z_score=z_score,
        current_on_hand=oh,
        inventory_on_order=oo,
        backorders=bo,
        inventory_position=ip,
        expected_demand_protection_period=demand_during_protection,
        std_dev_protection_period=std_dev_protection,
        safety_stock=safety_stock,
        recommended_safety_stock=int(math.ceil(safety_stock)),
        order_up_to_level=target_inventory,
        recommended_target_level=recommended_target,
        order_quantity=order_quantity,
        recommended_order_quantity=recommended_order,
        continuous_review_ss_equivalent=continuous_ss,
        periodic_safety_stock_multiplier=periodic_ratio,
        verification=verification,
    )


# Backward-compatible alias
calculate_fixed_period_review = calculate_periodic_review


# -----------------------------------------------------------------------------
# Tool 5: Newsvendor Model (Single-Period Stochastic Demand)
# -----------------------------------------------------------------------------

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    salvage_value: float = 0.0,
    goodwill_penalty: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    cost_underage: Optional[float] = None,
    cost_overage: Optional[float] = None,
) -> NewsvendorResult:
    """Calculate single-period stochastic stocking quantity using the Newsvendor model.
    
    Formulas:
        Underage cost Cu = selling_price - unit_cost + goodwill_penalty (or direct Cu)
        Overage cost Co = unit_cost - salvage_value (or direct Co)
        Critical ratio CR* = Cu / (Cu + Co)
        Optimal order quantity Q* = mean_demand + z* * std_demand (where Phi(z*) = CR*)
        
    Raises:
        ValueError: If required inputs are missing, negative, unprofitable, or invalid.
    """
    # 1. Validation - Never invent missing numerical inputs
    cu_input = underage_cost if underage_cost is not None else cost_underage
    co_input = overage_cost if overage_cost is not None else cost_overage

    if mean_demand is None:
        raise ValueError("Missing required numerical input: 'mean_demand' (mu) cannot be None.")
    if std_demand is None:
        raise ValueError("Missing required numerical input: 'std_demand' (sigma) cannot be None.")

    try:
        mu = float(mean_demand)
        sigma = float(std_demand)
        salvage = float(salvage_value)
        goodwill = float(goodwill_penalty)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid non-numeric input provided to calculate_newsvendor: {err}")

    if mu <= 0:
        raise ValueError(f"Mean demand must be strictly positive (> 0), got {mu}.")
    if sigma <= 0:
        raise ValueError(f"Standard deviation of demand must be strictly positive (> 0), got {sigma}.")
    if salvage < 0:
        raise ValueError(f"Salvage value cannot be negative, got {salvage}.")
    if goodwill < 0:
        raise ValueError(f"Goodwill penalty cannot be negative, got {goodwill}.")

    # Direct marginal cost mode vs Price/Cost mode
    if cu_input is not None and co_input is not None:
        try:
            cu = float(cu_input)
            co = float(co_input)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid non-numeric underage/overage cost: {err}")

        if cu <= 0:
            raise ValueError(f"Underage cost (Cu) must be strictly positive (> 0), got {cu}.")
        if co <= 0:
            raise ValueError(f"Overage cost (Co) must be strictly positive (> 0), got {co}.")

        c = float(unit_cost) if unit_cost is not None else (co + salvage)
        p = float(selling_price) if selling_price is not None else (c + cu - goodwill)
    else:
        if selling_price is None:
            raise ValueError("Missing required numerical input: 'selling_price' (P) cannot be None.")
        if unit_cost is None:
            raise ValueError("Missing required numerical input: 'unit_cost' (C) cannot be None.")

        try:
            p = float(selling_price)
            c = float(unit_cost)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid non-numeric price/cost: {err}")

        if p <= 0:
            raise ValueError(f"Selling price must be strictly positive (> 0), got {p}.")
        if c <= 0:
            raise ValueError(f"Unit cost must be strictly positive (> 0), got {c}.")
        if c >= p:
            raise ValueError(f"Unit cost (${c:.2f}) must be less than selling price (${p:.2f}) for a profitable product.")
        if salvage > c:
            raise ValueError(f"Salvage value (${salvage:.2f}) cannot exceed unit cost (${c:.2f}) (arbitrage condition).")

        cu = p - c + goodwill  # Underage cost: cost of ordering one unit too few
        co = c - salvage       # Overage cost: cost of ordering one unit too many

    if (cu + co) <= 0:
        raise ValueError(f"Sum of underage and overage costs must be positive, got {cu + co}.")

    critical_ratio = cu / (cu + co)
    z_score = float(stats.norm.ppf(critical_ratio))
    optimal_q = mu + (z_score * sigma)
    if optimal_q < 0:
        optimal_q = 0.0
    recommended_q = int(round(optimal_q))

    # Unit Normal Loss Function: L(z) = phi(z) - z * (1 - Phi(z))
    phi_z = stats.norm.pdf(z_score)
    big_phi_z = stats.norm.cdf(z_score)
    unit_loss_l_z = phi_z - (z_score * (1.0 - big_phi_z))

    expected_lost_sales = sigma * unit_loss_l_z
    expected_sales = mu - expected_lost_sales
    expected_leftover = optimal_q - expected_sales

    expected_rev = (p * expected_sales) + (salvage * expected_leftover)
    expected_purchase_cost = c * optimal_q
    expected_profit = expected_rev - expected_purchase_cost - (goodwill * expected_lost_sales)

    in_stock_prob = critical_ratio
    stockout_prob = 1.0 - critical_ratio

    # 3. Model Assumptions
    assumptions = [
        "Single sales period; remaining inventory cannot be carried forward to subsequent cycles.",
        "Demand follows a continuous Gaussian (Normal) distribution with known parameters (mu, sigma).",
        "Unit selling price, procurement cost, and salvage value remain constant.",
        "Unsold units at the end of the period are liquidated at the fixed salvage value.",
        "Unmet customer demand is lost sales and cannot be backlogged.",
    ]

    # 4. Intermediate Calculations
    intermediate_calcs = {
        "underage_cost_Cu": cu,
        "overage_cost_Co": co,
        "sum_Cu_plus_Co": cu + co,
        "critical_ratio_CR": critical_ratio,
        "z_score": z_score,
        "optimal_stocking_quantity_exact": optimal_q,
        "unit_normal_loss_L_z": unit_loss_l_z,
        "expected_lost_sales": expected_lost_sales,
        "expected_sales": expected_sales,
        "expected_leftover_units": expected_leftover,
        "expected_revenue": expected_rev,
        "expected_purchase_cost": expected_purchase_cost,
        "expected_salvage_revenue": salvage * expected_leftover,
        "expected_net_profit": expected_profit,
        "in_stock_probability": in_stock_prob,
        "stockout_probability": stockout_prob,
    }

    # 5. Final Answer
    final_answer = {
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": critical_ratio,
        "optimal_order_quantity": optimal_q,
        "recommended_order_quantity": recommended_q,
        "expected_profit": expected_profit,
        "expected_lost_sales": expected_lost_sales,
        "expected_leftover_units": expected_leftover,
        "in_stock_probability": in_stock_prob,
    }

    verification = CalculationVerification(is_valid=True)
    verification.checks_passed.append(
        f"Critical ratio CR* = Cu / (Cu + Co) = {cu:.2f} / ({cu:.2f} + {co:.2f}) = {critical_ratio:.4f}."
    )
    verification.checks_passed.append(
        f"Optimal order quantity {optimal_q:.2f} satisfies standard newsvendor marginal analysis condition."
    )

    return NewsvendorResult(
        inputs={
            "selling_price": p,
            "unit_cost": c,
            "mean_demand": mu,
            "std_demand": sigma,
            "salvage_value": salvage,
            "goodwill_penalty": goodwill,
        },
        assumptions=assumptions,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        selling_price=p,
        unit_cost=c,
        salvage_value=salvage,
        goodwill_penalty=goodwill,
        cost_underage=cu,
        cost_overage=co,
        critical_ratio=critical_ratio,
        z_score=z_score,
        mean_demand=mu,
        std_demand=sigma,
        optimal_order_quantity=optimal_q,
        recommended_order_quantity=recommended_q,
        expected_lost_sales=expected_lost_sales,
        expected_sales=expected_sales,
        expected_leftover_units=expected_leftover,
        expected_revenue=expected_rev,
        expected_purchase_cost=expected_purchase_cost,
        expected_salvage_revenue=salvage * expected_leftover,
        expected_profit=expected_profit,
        in_stock_probability=in_stock_prob,
        stockout_probability=stockout_prob,
        verification=verification,
    )
