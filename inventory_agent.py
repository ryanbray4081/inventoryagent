"""Supply Chain Analytics Inventory Decision Agent.

This module implements the complete 7-stage agent pipeline:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

All numerical calculations are executed deterministically by Python tools in inventory_tools.py.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from inventory_tools import (
    CalculationVerification,
    EOQResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
    PeriodicReviewResult,
    QuantityDiscountResult,
    ROPResult,
    SafetyStockROPResult,
    calculate_eoq,
    calculate_fixed_period_review,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
    calculate_safety_stock_rop,
)


class ModelType(str, Enum):
    EOQ = "EOQ"
    QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    NEWSVENDOR = "NEWSVENDOR"
    UNKNOWN = "UNKNOWN"


MODEL_DISPLAY_NAMES = {
    ModelType.EOQ: "Economic Order Quantity (EOQ)",
    ModelType.QUANTITY_DISCOUNT: "Quantity Discount Analysis",
    ModelType.SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (SS & ROP)",
    ModelType.FIXED_PERIOD_REVIEW: "Fixed-Period Review (Order-Up-To Level)",
    ModelType.NEWSVENDOR: "Newsvendor Single-Period Model",
    ModelType.UNKNOWN: "Unrecognized Model",
}

TOOL_MAPPING = {
    ModelType.EOQ: "inventory_tools.calculate_eoq",
    ModelType.QUANTITY_DISCOUNT: "inventory_tools.calculate_quantity_discount",
    ModelType.SAFETY_STOCK_ROP: "inventory_tools.calculate_rop",
    ModelType.FIXED_PERIOD_REVIEW: "inventory_tools.calculate_periodic_review",
    ModelType.NEWSVENDOR: "inventory_tools.calculate_newsvendor",
}


@dataclass
class Stage1Question:
    raw_question: str


@dataclass
class Stage2ModelIdentification:
    model_type: ModelType
    model_name: str
    confidence: float
    rationale: str
    alternative_candidates: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class Stage3ExtractedInputs:
    parameters: Dict[str, Any]
    defaults_applied: Dict[str, Any]
    missing_critical: List[str]
    notes: List[str]


@dataclass
class Stage4ToolSelection:
    tool_function: str
    arguments: Dict[str, Any]


@dataclass
class Stage5CalculationResult:
    result_object: Any
    result_dict: Dict[str, Any]


@dataclass
class Stage6Verification:
    is_valid: bool
    checks_passed: List[str]
    warnings: List[str]
    errors: List[str]


@dataclass
class Stage7Explanation:
    executive_summary: str
    detailed_explanation: str
    actionable_recommendations: List[str]
    markdown_report: str


@dataclass
class AgentPipelineTrace:
    question: str
    model_identification: Stage2ModelIdentification
    extracted_inputs: Stage3ExtractedInputs
    tool_selection: Stage4ToolSelection
    calculation: Optional[Stage5CalculationResult]
    verification: Stage6Verification
    explanation: Stage7Explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "model_identification": asdict(self.model_identification),
            "extracted_inputs": asdict(self.extracted_inputs),
            "tool_selection": asdict(self.tool_selection),
            "calculation": self.calculation.result_dict if self.calculation else None,
            "verification": asdict(self.verification),
            "explanation": asdict(self.explanation),
        }


class InventoryDecisionAgent:
    """Intelligent agent orchestrating inventory decision analysis."""

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # STAGE 1 & 2: Identify Model from Customer Question
    # -------------------------------------------------------------------------

    def identify_model(self, question: str) -> Stage2ModelIdentification:
        """Classify question into one of the 5 inventory models."""
        q_lower = question.lower()
        # Normalized text with spaces replacing hyphens and underscores for keyword matching
        q_norm = re.sub(r"[-_]", " ", q_lower)

        scores = {
            ModelType.EOQ: 0.0,
            ModelType.QUANTITY_DISCOUNT: 0.0,
            ModelType.SAFETY_STOCK_ROP: 0.0,
            ModelType.FIXED_PERIOD_REVIEW: 0.0,
            ModelType.NEWSVENDOR: 0.0,
        }

        # Quantity Discount keywords (high precedence when discount or tiers mentioned)
        if any(w in q_norm for w in ["tier", "discount", "price break", "volume discount", "bulk discount", "bracket"]):
            scores[ModelType.QUANTITY_DISCOUNT] += 5.0
        if re.search(r"\$\d+.*(?:\$|\b)\d+.*tier", q_norm) or re.search(r"\d+\s*[-–]\s*\d+", question):
            scores[ModelType.QUANTITY_DISCOUNT] += 2.5

        # Fixed-Period Review keywords (Periodic Review / Order-Up-To / Base Stock)
        if any(w in q_norm for w in ["fixed period", "periodic review", "order up to", "target stock", "base stock", "review period", "every week", "every 7 days", "every 14 days", "every month", "weekly review", "monthly review"]):
            scores[ModelType.FIXED_PERIOD_REVIEW] += 6.0
        if any(w in q_norm for w in ["on hand", "current inventory", "inventory position", "on order"]):
            scores[ModelType.FIXED_PERIOD_REVIEW] += 2.0

        # Newsvendor keywords
        if any(w in q_norm for w in ["newsvendor", "newsboy", "single period", "salvage", "scrap value", "perishable", "seasonal", "goodwill", "critical fractile", "critical ratio", "underage", "overage"]):
            scores[ModelType.NEWSVENDOR] += 6.0
        if any(w in q_norm for w in ["selling price", "retail price"]) and any(w in q_norm for w in ["salvage", "leftover", "fashion", "holiday", "winter"]):
            scores[ModelType.NEWSVENDOR] += 3.0

        # Safety Stock & ROP keywords
        if any(w in q_norm for w in ["reorder point", "rop", "safety stock", "cycle service level", "lead time demand"]):
            scores[ModelType.SAFETY_STOCK_ROP] += 5.0
        elif "service level" in q_norm:
            scores[ModelType.SAFETY_STOCK_ROP] += 2.5
        if any(w in q_norm for w in ["lead time", "supplier lead"]) and any(w in q_norm for w in ["standard deviation", "std dev", "uncertainty", "variance", "sigma"]):
            scores[ModelType.SAFETY_STOCK_ROP] += 3.0

        # EOQ keywords
        if any(w in q_norm for w in ["eoq", "economic order quantity", "optimal lot size", "order size"]):
            scores[ModelType.EOQ] += 4.5
        if any(w in q_norm for w in ["holding cost", "carrying cost"]) and any(w in q_norm for w in ["ordering cost", "setup cost", "order cost"]):
            scores[ModelType.EOQ] += 3.5
        if any(w in q_norm for w in ["annual demand", "orders per year", "order cycle"]):
            scores[ModelType.EOQ] += 2.0

        # Disambiguation penalties/boosts:
        # If periodic review or order-up-to is detected, suppress ROP because ROP is often part of service level description
        if any(w in q_norm for w in ["review period", "periodic review", "order up to", "target stock", "base stock", "weekly review", "monthly review"]):
            scores[ModelType.FIXED_PERIOD_REVIEW] += 3.0
            scores[ModelType.SAFETY_STOCK_ROP] = max(0.0, scores[ModelType.SAFETY_STOCK_ROP] - 4.0)

        # If discount tiers exist, prefer QUANTITY_DISCOUNT over plain EOQ
        if scores[ModelType.QUANTITY_DISCOUNT] > 2.0 and scores[ModelType.EOQ] > 0:
            scores[ModelType.QUANTITY_DISCOUNT] += 2.0

        # Pick top score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_model, top_score = sorted_candidates[0]

        if top_score < 1.0:
            # Fallback heuristic
            if "order" in q_lower and ("cost" in q_lower or "demand" in q_lower):
                top_model = ModelType.EOQ
                top_score = 1.0
            else:
                top_model = ModelType.UNKNOWN
                top_score = 0.0

        confidence = min(0.98, max(0.40, top_score / 6.0)) if top_model != ModelType.UNKNOWN else 0.10

        rationale_map = {
            ModelType.EOQ: "Identified standard continuous deterministic lot-sizing problem balancing ordering cost against inventory carrying/holding cost.",
            ModelType.QUANTITY_DISCOUNT: "Identified multi-tier unit pricing schedule requiring all-units or incremental discount evaluation across price breaks.",
            ModelType.SAFETY_STOCK_ROP: "Identified inventory buffer determination under demand or lead time uncertainty to satisfy target cycle service level.",
            ModelType.FIXED_PERIOD_REVIEW: "Identified periodic inspection system (R, S policy) requiring Order-Up-To Level and protection interval (R+L) netting.",
            ModelType.NEWSVENDOR: "Identified single-period stochastic decision balancing marginal cost of understocking against overstocking for perishable/seasonal goods.",
            ModelType.UNKNOWN: "Could not decisively map question to a known inventory policy.",
        }

        alt_candidates = [(MODEL_DISPLAY_NAMES[m], round(s, 2)) for m, s in sorted_candidates[1:3] if s > 0]

        return Stage2ModelIdentification(
            model_type=top_model,
            model_name=MODEL_DISPLAY_NAMES[top_model],
            confidence=round(confidence, 2),
            rationale=rationale_map.get(top_model, ""),
            alternative_candidates=alt_candidates,
        )

    # -------------------------------------------------------------------------
    # STAGE 3: Extract Inputs from Natural Language
    # -------------------------------------------------------------------------

    def extract_inputs(self, question: str, model_type: ModelType) -> Stage3ExtractedInputs:
        """Extract typed parameters with smart unit conversions, synonyms, and defaults.
        
        Examines the entire question across multiple syntactic forms (keyword-before-number,
        number-before-keyword, colon/equals notation, abbreviations) before deciding whether
        any parameter is missing.
        """
        # Normalize text: strip commas inside numbers (e.g. 12,000 -> 12000, 100,000 -> 100000)
        q = re.sub(r"(?<=\d),(?=\d)", "", question)
        params: Dict[str, Any] = {}
        defaults: Dict[str, Any] = {}
        missing: List[str] = []
        notes: List[str] = []

        def find_number(patterns: List[str], text: str) -> Optional[float]:
            """Search list of regex patterns and return first valid float match."""
            for p in patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    for grp in m.groups():
                        if grp is not None:
                            try:
                                clean_val = grp.replace("$", "").replace(",", "").strip()
                                return float(clean_val)
                            except ValueError:
                                continue
            return None

        # =====================================================================
        # MODEL 1: ECONOMIC ORDER QUANTITY (EOQ)
        # =====================================================================
        if model_type == ModelType.EOQ:
            # 1. Demand (D): Annual, monthly, or daily
            # Check explicitly for annual demand first
            demand = find_number([
                r"(?:annual\s*demand|yearly\s*demand|demand\s*per\s*year|annual\s*requirement|annual\s*usage)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:annual\s*(?:demand|requirement|usage|units)|yearly\s*(?:demand|requirement|usage)|units\s*per\s*year|units/year|annual\s*units|units\s*annually|units\s*per\s*annum)",
                r"(?:^|[\s,;(])D\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # Check for monthly demand conversion if annual not directly found
            if demand is None:
                m_dem = find_number([
                    r"(?:monthly\s*demand|demand\s*per\s*month)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                    r"(\d+(?:\.\d+)?)\s*(?:units\s*per\s*month|units/month|monthly\s*units)",
                ], q)
                if m_dem is not None:
                    demand = m_dem * 12.0
                    notes.append(f"Converted monthly demand ({m_dem:,.0f} units/mo) to annual demand ({demand:,.0f} units/yr).")

            # Check for daily demand conversion if still not found
            if demand is None:
                d_dem = find_number([
                    r"(?:daily\s*demand|demand\s*per\s*day)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                    r"(\d+(?:\.\d+)?)\s*(?:units\s*per\s*day|units/day|daily\s*units)",
                ], q)
                if d_dem is not None:
                    demand = d_dem * 250.0
                    notes.append(f"Converted daily demand ({d_dem:,.0f} units/day) to annual demand ({demand:,.0f} units/yr) assuming 250 work days.")

            # Generic demand fallback if neither annual nor monthly/daily keywords were found
            if demand is None:
                demand = find_number([
                    r"(?:demand\s*of|demand\s*is|demand:?)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                    r"demand\s*(?:=|:)?\s*(\d+(?:\.\d+)?)",
                ], q)

            # 2. Ordering / Setup Cost (S)
            # Synonyms: ordering cost, order cost, setup cost, setup expense, cost per order,
            # cost of placing an order, ordering/setup cost, order fee, setup fee, reorder cost
            order_cost = find_number([
                # Keyword before number
                r"(?:ordering\s*(?:and|/)?\s*setup\s*cost|ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|cost\s*per\s*order|cost\s*of\s*placing\s*(?:an\s*)?order|cost\s*to\s*place\s*(?:an\s*)?order|ordering\s*expense|order\s*expense|order\s*fee|setup\s*fee|reorder\s*cost)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                # Number before keyword
                r"\$?(\d+(?:\.\d+)?)\s*(?:ordering\s*(?:and|/)?\s*setup\s*cost|ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|cost\s*per\s*order|cost\s*of\s*placing\s*(?:an\s*)?order|cost\s*to\s*place\s*(?:an\s*)?order|order\s*fee|setup\s*fee|per\s*order|to\s*place\s*(?:an\s*)?order|per\s*setup)",
                # Symbolic notation S = X
                r"(?:^|[\s,;(])S\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 3. Holding / Carrying Cost (H)
            # Synonyms: holding cost, carrying cost, holding expense, carrying expense, cost of carrying, cost to hold
            holding_cost = find_number([
                r"(?:holding\s*cost|carrying\s*cost|holding\s*expense|carrying\s*expense|cost\s*of\s*carrying|cost\s*to\s*hold|storage\s*cost|inventory\s*holding\s*cost)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)(?!\s*%)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:holding\s*cost|carrying\s*cost|holding\s*expense|carrying\s*expense)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit\s*per\s*year|/unit/year|per\s*unit\s*annually|per\s*unit\s*carrying\s*cost|per\s*unit\s*holding\s*cost)",
                r"(?:^|[\s,;(])H\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 4. Unit purchase price (C)
            unit_cost = find_number([
                r"(?:unit\s*cost|purchase\s*cost|unit\s*price|item\s*cost|wholesale\s*cost|procurement\s*cost|unit\s*purchase\s*price)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"(?:cost\s*per\s*unit|price\s*per\s*unit)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)(?!\s*per\s*year)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:each|apiece|per\s*unit)(?!\s*per\s*year)",
                r"(?:^|[\s,;(])C\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 5. Percentage holding rate (i)
            holding_rate = find_number([
                r"(?:holding\s*rate|carrying\s*rate|annual\s*holding\s*rate|annual\s*carrying\s*charge|percentage\s*holding\s*cost|holding\s*percentage)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:unit\s*)?price|of\s*item\s*cost|holding|carrying)",
                r"(?:^|[\s,;(])i\s*(?:=|:)\s*(\d+(?:\.\d+)?)\s*%",
            ], q)

            # Derive holding cost from rate * unit_cost if holding cost not given explicitly
            if holding_cost is None and holding_rate is not None and unit_cost is not None:
                rate_val = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
                holding_cost = rate_val * unit_cost
                notes.append(f"Derived holding cost ${holding_cost:.2f} from {holding_rate}% holding rate applied to ${unit_cost:.2f} unit price.")

            work_days = find_number([
                r"(\d+)\s*(?:work\s*days|working\s*days|business\s*days|operating\s*days)",
            ], q)

            if demand is not None:
                params["demand"] = demand
            else:
                missing.append("annual demand (D)")

            if order_cost is not None:
                params["ordering_cost"] = order_cost
            else:
                missing.append("ordering cost (S)")

            if holding_cost is not None:
                params["holding_cost"] = holding_cost
            else:
                missing.append("holding cost (H)")

            params["unit_cost"] = unit_cost if unit_cost is not None else 0.0
            if work_days is not None:
                params["work_days_per_year"] = int(work_days)
            else:
                defaults["work_days_per_year"] = 250
                params["work_days_per_year"] = 250

        # =====================================================================
        # MODEL 2: QUANTITY DISCOUNT ANALYSIS
        # =====================================================================
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            # 1. Demand
            demand = find_number([
                r"(?:annual\s*demand|yearly\s*demand|demand\s*per\s*year|demand\s*of|demand\s*is|demand:?)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*per\s*year|units/year|annual\s*units)",
                r"(?:^|[\s,;(])D\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
                r"demand\s*(?:=|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # 2. Ordering Cost (all EOQ ordering cost synonyms)
            order_cost = find_number([
                r"(?:ordering\s*(?:and|/)?\s*setup\s*cost|ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|cost\s*per\s*order|cost\s*of\s*placing\s*(?:an\s*)?order|cost\s*to\s*place\s*(?:an\s*)?order|order\s*fee|setup\s*fee|reorder\s*cost)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:ordering\s*(?:and|/)?\s*setup\s*cost|ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|cost\s*per\s*order|cost\s*of\s*placing\s*(?:an\s*)?order|per\s*order)",
                r"(?:^|[\s,;(])S\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 3. Percentage Holding Cost
            holding_rate = find_number([
                r"(?:percentage\s*holding\s*cost|holding\s*rate|carrying\s*rate|holding\s*cost\s*is|carrying\s*rate\s*is|annual\s*carrying\s*rate)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:holding|carrying|of\s*price|of\s*unit\s*price|carrying\s*charge)",
            ], q)

            # 4. Fixed Holding Cost
            fixed_holding_cost = find_number([
                r"(?:fixed\s*holding\s*cost|fixed\s*carrying\s*cost|holding\s*cost\s*of|carrying\s*cost\s*of)\s*(?:is|:|=)?\s*\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit|/unit)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit\s*per\s*year|/unit/year|per\s*unit\s*holding\s*cost)",
            ], q)

            # 5. Quantity Breaks & Unit Prices (Tiers)
            tiers: List[Dict[str, Any]] = []

            # Range tier: 1-499 units at $10.00, 1 to 499 for $10.00, etc.
            range_matches = re.findall(
                r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*(?:units)?\s*(?:at|for|:|=|\$|costs?|priced\s*at)?\s*\$?(\d+(?:\.\d+)?)",
                q,
                re.IGNORECASE,
            )
            for m in range_matches:
                tiers.append({
                    "min_qty": float(m[0]),
                    "max_qty": float(m[1]),
                    "unit_price": float(m[2]),
                })

            # Open-ended upper tier: 1000+ units at $8.00, 1000 and above for $8.00, etc.
            plus_matches = re.findall(
                r"(\d+)\s*(?:\+|and\s*above|or\s*more|and\s*over|and\s*up)\s*(?:units)?\s*(?:at|for|:|=|\$|costs?|priced\s*at)?\s*\$?(\d+(?:\.\d+)?)",
                q,
                re.IGNORECASE,
            )
            for m in plus_matches:
                tiers.append({
                    "min_qty": float(m[0]),
                    "max_qty": None,
                    "unit_price": float(m[1]),
                })

            # Single price break structure: "Price is $10 for under 500, and $8 for 500 or more"
            if not tiers:
                under_match = re.search(
                    r"(?:under|less\s*than)\s*(\d+)\s*(?:units)?\s*(?:costs?|is|at|for|:)?\s*\$?(\d+(?:\.\d+)?)",
                    q,
                    re.IGNORECASE,
                )
                over_match = re.search(
                    r"(?:over|at\s*least|from)\s*(\d+)\s*(?:units)?\s*(?:costs?|is|at|for|:)?\s*\$?(\d+(?:\.\d+)?)",
                    q,
                    re.IGNORECASE,
                )
                if under_match and over_match:
                    tiers.append({
                        "min_qty": 0.0,
                        "max_qty": float(under_match.group(1)) - 1.0,
                        "unit_price": float(under_match.group(2)),
                    })
                    tiers.append({
                        "min_qty": float(over_match.group(1)),
                        "max_qty": None,
                        "unit_price": float(over_match.group(2)),
                    })

            if not tiers:
                # Fallback to example discount schedule only if no tiers could be parsed
                tiers = [
                    {"min_qty": 0, "max_qty": 499, "unit_price": 10.0},
                    {"min_qty": 500, "max_qty": 999, "unit_price": 9.0},
                    {"min_qty": 1000, "max_qty": None, "unit_price": 8.0},
                ]
                defaults["tiers"] = "Example discount schedule: [0-499: $10, 500-999: $9, 1000+: $8]"
                notes.append("No explicit discount tiers extracted from query; loaded baseline 3-tier schedule.")

            if demand is not None:
                params["demand"] = demand
            else:
                missing.append("annual demand (D)")

            if order_cost is not None:
                params["ordering_cost"] = order_cost
            else:
                missing.append("ordering cost (S)")

            if holding_rate is not None:
                rate_norm = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
                params["holding_rate"] = rate_norm
            elif fixed_holding_cost is not None:
                params["fixed_holding_cost"] = fixed_holding_cost
            else:
                defaults["holding_rate"] = 0.20
                params["holding_rate"] = 0.20
                notes.append("Holding rate not specified; defaulted to standard 20% carrying rate of unit price.")

            params["tiers"] = tiers

        # =====================================================================
        # MODEL 3: SAFETY STOCK AND REORDER POINT (SS & ROP)
        # =====================================================================
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            # 1. Lead-Time Standard Deviation (sigma_L) - Extract FIRST to avoid collision
            std_lead_time = find_number([
                r"(?:lead[\s-]*time\s*standard\s*deviation|standard\s*deviation\s*of\s*lead[\s-]*time|lead[\s-]*time\s*std\s*dev|std\s*dev\s*of\s*lead[\s-]*time|lead[\s-]*time\s*variability|lead[\s-]*time\s*uncertainty|sigma_l|sigma_lt)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:days?\s*)?(?:lead[\s-]*time\s*standard\s*deviation|lead[\s-]*time\s*std\s*dev|std\s*dev\s*of\s*lead[\s-]*time)",
                r"lead[\s-]*time\s*varies\s*(?:by|with\s*std\s*dev\s*of)?\s*(\d+(?:\.\d+)?)",
                r"(?:^|[\s,;(])(?:sigma_l|sigma_lt)\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 2. Demand Standard Deviation (sigma_d)
            std_demand = find_number([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|demand\s*std\s*dev|std\s*dev\s*of\s*demand|daily\s*demand\s*std\s*dev|daily\s*std\s*dev|demand\s*variability|sigma_d)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units?\s*)?(?:demand\s*standard\s*deviation|demand\s*std\s*dev|std\s*dev\s*of\s*demand|standard\s*deviation\s*of\s*demand)",
                r"demand\s*varies\s*(?:by|with\s*std\s*dev\s*of)?\s*(\d+(?:\.\d+)?)",
                r"(?:^|[\s,;(])sigma_d\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # Generic standard deviation fallback (excluding lead-time context without illegal lookbehind)
            if std_demand is None:
                for m in re.finditer(r"(?:standard\s*deviation|std\s*dev|sigma)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)", q, re.IGNORECASE):
                    pos = m.start()
                    preceding = q[max(0, pos - 30):pos].lower()
                    following = q[m.end():min(len(q), m.end() + 20)].lower()
                    if "lead" not in preceding and "lead" not in following and "day" not in following:
                        try:
                            std_demand = float(m.group(1))
                            break
                        except ValueError:
                            pass

            # 3. Average Lead Time (L)
            lead_time = find_number([
                r"(?:average\s*lead[\s-]*time|mean\s*lead[\s-]*time|supplier\s*lead[\s-]*time|delivery\s*lead[\s-]*time|replenishment\s*lead[\s-]*time|delivery\s*time|lead[\s-]*time)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:days?|business\s*days?|operating\s*days?|weeks?)?",
                r"(\d+(?:\.\d+)?)\s*(?:days?|business\s*days?|operating\s*days?)\s*(?:supplier\s*)?lead[\s-]*time",
                r"takes\s*(\d+(?:\.\d+)?)\s*days?\s*to\s*(?:arrive|deliver|replenish)",
                r"(?:^|[\s,;(])L\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 4. Average Demand (d)
            daily_demand = find_number([
                r"(?:average\s*daily\s*demand|mean\s*daily\s*demand|daily\s*demand|daily\s*usage|daily\s*sales|average\s*demand|mean\s*demand|demand\s*rate)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units?\s*)?(?:per\s*day|/day|daily\s*demand|daily\s*usage|daily\s*sales)",
                r"demand\s*(?:is|of|=|:)\s*(\d+(?:\.\d+)?)\s*(?:units?\s*per\s*day|units?/day|units?\s*daily)",
                r"(?:^|[\s,;(])d\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # Fallback to annual demand conversion if daily demand not specified
            if daily_demand is None:
                ann_dem = find_number([
                    r"(?:annual\s*demand|demand\s*per\s*year)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                    r"(\d+(?:\.\d+)?)\s*(?:annual\s*(?:demand|requirement|usage|units)|units\s*per\s*year|units/year)",
                ], q)
                if ann_dem is not None:
                    daily_demand = ann_dem / 250.0
                    notes.append(f"Derived daily demand ({daily_demand:.2f} units/day) from annual demand ({ann_dem:,.0f} units/yr) assuming 250 operating days.")

            # 5. Service Level (CSL)
            service_level = find_number([
                r"(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|csl|in-stock\s*probability|fill\s*rate)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*%?",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s*service\s*level|service\s*level|csl|in-stock)",
                r"(0\.\d+)\s*(?:cycle\s*service\s*level|service\s*level|csl)",
            ], q)

            unit_cost = find_number([
                r"(?:unit\s*cost|purchase\s*cost|item\s*cost|unit\s*price)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            holding_rate = find_number([
                r"(?:holding\s*rate|carrying\s*rate)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*%",
            ], q)

            if daily_demand is not None:
                params["demand"] = daily_demand
            else:
                missing.append("average demand (d)")

            if lead_time is not None:
                params["lead_time"] = lead_time
            else:
                missing.append("lead time (L)")

            params["std_demand"] = std_demand if std_demand is not None else 0.0
            params["std_lead_time"] = std_lead_time if std_lead_time is not None else 0.0

            if service_level is not None:
                params["service_level"] = service_level / 100.0 if service_level > 1.0 else service_level
            else:
                defaults["service_level"] = 0.95
                params["service_level"] = 0.95
                notes.append("Service level not provided; defaulted to standard 95% Cycle Service Level (CSL).")

            params["unit_cost"] = unit_cost if unit_cost is not None else 0.0
            params["annual_holding_rate"] = (holding_rate / 100.0) if (holding_rate and holding_rate > 1.0) else (holding_rate if holding_rate else 0.20)

        # =====================================================================
        # MODEL 4: FIXED-PERIOD REVIEW (ORDER-UP-TO LEVEL)
        # =====================================================================
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            # 1. Review Period (R)
            review_period = find_number([
                r"(?:review\s*period|reviewed\s*every|review\s*every|periodic\s*review\s*every|periodic\s*review\s*of|review\s*cycle|inspection\s*period|inspection\s*every|order\s*placed\s*every)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"every\s*(\d+(?:\.\d+)?)\s*(?:days|business\s*days|operating\s*days)",
                r"(\d+(?:\.\d+)?)\s*(?:days?\s*review|day\s*review\s*period|days?\s*periodic\s*review|days?\s*review\s*cycle)",
                r"(?:^|[\s,;(])R\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            if review_period is None:
                if "weekly" in q.lower() or "every week" in q.lower():
                    review_period = 7.0
                    notes.append("Extracted 7 days review period from 'weekly'.")
                elif "bi-weekly" in q.lower() or "every two weeks" in q.lower() or "every 2 weeks" in q.lower():
                    review_period = 14.0
                    notes.append("Extracted 14 days review period from 'bi-weekly'.")
                elif "monthly" in q.lower() or "every month" in q.lower():
                    review_period = 30.0
                    notes.append("Extracted 30 days review period from 'monthly'.")

            # 2. Lead Time (L)
            lead_time = find_number([
                r"(?:lead\s*time|supplier\s*lead\s*time|delivery\s*time|replenishment\s*lead\s*time)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:days?\s*lead\s*time|day\s*lead\s*time)",
                r"(?:^|[\s,;(])L\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 3. Demand (d)
            daily_demand = find_number([
                r"(?:daily\s*demand|average\s*daily\s*demand|mean\s*daily\s*demand|demand\s*of|demand\s*is)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*per\s*day|units/day)",
                r"(?:^|[\s,;(])d\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 4. Standard Deviation of Demand (sigma_d)
            std_demand = find_number([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|std\s*dev\s*of\s*demand|demand\s*std\s*dev|std\s*dev|standard\s*deviation|sigma)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units?\s*)?(?:std\s*dev|standard\s*deviation)",
                r"(?:^|[\s,;(])sigma_d\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 5. Service Level
            service_level = find_number([
                r"(?:service\s*level|cycle\s*service\s*level|csl|target\s*service\s*level)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*%?",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:service\s*level|csl)",
                r"(0\.\d+)\s*(?:service\s*level|csl)",
            ], q)

            # 6. Inventory Position (IP) directly specified
            inv_position = find_number([
                r"(?:inventory\s*position|effective\s*inventory\s*position)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(?:^|[\s,;(])IP\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 7. On-Hand Inventory
            on_hand = find_number([
                r"(?:on-hand\s*inventory|on\s*hand|current\s*inventory|current\s*stock|currently\s*have)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*on\s*hand|units\s*in\s*stock)",
            ], q)

            # 8. On-Order Inventory
            on_order = find_number([
                r"(?:on-order\s*inventory|on\s*order|pipeline\s*inventory|already\s*ordered|in\s*transit)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*on\s*order|units\s*in\s*transit)",
            ], q)

            # 9. Backorders
            backorders = find_number([
                r"(?:backorder|backorders|backordered|unfilled\s*orders|backlog)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*backordered|backorders)",
            ], q)

            if review_period is not None:
                params["review_period"] = review_period
            else:
                missing.append("review period length (R)")

            if lead_time is not None:
                params["lead_time"] = lead_time
            else:
                missing.append("lead time (L)")

            if daily_demand is not None:
                params["demand"] = daily_demand
            else:
                missing.append("demand per period (d)")

            params["std_demand"] = std_demand if std_demand is not None else 0.0

            if service_level is not None:
                params["service_level"] = service_level / 100.0 if service_level > 1.0 else service_level
            else:
                defaults["service_level"] = 0.95
                params["service_level"] = 0.95
                notes.append("Service level defaulted to 95%.")

            if inv_position is not None:
                params["inventory_position"] = inv_position
                notes.append(f"Used directly specified inventory position ({inv_position:,.0f} units).")
            else:
                params["current_inventory"] = on_hand if on_hand is not None else 0.0
                params["inventory_on_order"] = on_order if on_order is not None else 0.0
                params["backorders"] = backorders if backorders is not None else 0.0

        # =====================================================================
        # MODEL 5: NEWSVENDOR MODEL (SINGLE-PERIOD STOCHASTIC)
        # =====================================================================
        elif model_type == ModelType.NEWSVENDOR:
            # 1. Selling Price (P)
            price = find_number([
                r"(?:selling\s*price|retail\s*price|sales\s*price|price\s*per\s*unit|sell\s*for|sells\s*for|sold\s*for)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:retail\s*price|selling\s*price|retail|sales\s*price)",
                r"(?:^|[\s,;(])P\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 2. Purchase Cost (C)
            cost = find_number([
                r"(?:purchase\s*cost|wholesale\s*cost|wholesale\s*price|procurement\s*cost|unit\s*cost|production\s*cost|manufacturing\s*cost|costs?\s*to\s*(?:buy|make|produce))\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:unit\s*cost|wholesale|purchase\s*cost|procurement\s*cost|to\s*purchase|to\s*make)",
                r"(?:^|[\s,;(])C\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 3. Salvage Value (Sv)
            salvage = find_number([
                r"(?:salvage\s*value|scrap\s*value|liquidation\s*value|salvaged\s*at|liquidated\s*for|salvaged\s*for|markdown\s*price|residual\s*value|salvage)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:salvage|scrap\s*value|liquidation\s*value)",
                r"(?:^|[\s,;(])(?:Sv|S_v)\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 4. Underage / Shortage Cost (Cu) - Direct specification
            underage = find_number([
                r"(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage|cost\s*of\s*understocking|stockout\s*cost|lost\s*profit\s*per\s*unit|shortage\s*penalty)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage)",
                r"(?:^|[\s,;(])(?:Cu|C_u)\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 5. Overage / Excess Cost (Co) - Direct specification
            overage = find_number([
                r"(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage|cost\s*of\s*overstocking|overstock\s*cost)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage)",
                r"(?:^|[\s,;(])(?:Co|C_o)\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 6. Goodwill Penalty / Shortage Penalty
            goodwill = find_number([
                r"(?:goodwill\s*penalty|loss\s*of\s*goodwill|penalty\s*cost|shortage\s*penalty)\s*(?:of|is|:|=)?\s*\$?(\d+(?:\.\d+)?)",
                r"(?:^|[\s,;(])g\s*(?:=|:)\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # 7. Mean Demand (mu)
            mean_d = find_number([
                r"(?:mean\s*demand|average\s*demand|expected\s*demand|demand\s*mean|forecasted\s*demand)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(?:^|[\s,;(])(?:mean|mu)\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # 8. Demand Standard Deviation (sigma)
            std_d = find_number([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|demand\s*std\s*dev|std\s*dev\s*of\s*demand|std\s*dev|standard\s*deviation)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:demand\s*std\s*dev|standard\s*deviation)",
                r"(?:^|[\s,;(])(?:sigma|std_dev)\s*(?:=|:)\s*(\d+(?:\.\d+)?)",
            ], q)

            # Check if distribution family is explicitly mentioned
            if "normal" in q.lower() or "gaussian" in q.lower():
                notes.append("Detected Normal (Gaussian) demand distribution.")

            # Validate parameter completeness:
            # Either (selling_price AND unit_cost) OR (underage_cost AND overage_cost) must be present!
            if underage is not None and overage is not None:
                params["underage_cost"] = underage
                params["overage_cost"] = overage
                notes.append(f"Used directly specified marginal costs: Cu=${underage:.2f}, Co=${overage:.2f}.")
            else:
                if price is not None:
                    params["selling_price"] = price
                else:
                    missing.append("selling price (P)")

                if cost is not None:
                    params["unit_cost"] = cost
                else:
                    missing.append("unit purchase cost (C)")

            if mean_d is not None:
                params["mean_demand"] = mean_d
            else:
                missing.append("mean demand (mu)")

            if std_d is not None:
                params["std_demand"] = std_d
            else:
                missing.append("demand standard deviation (sigma)")

            params["salvage_value"] = salvage if salvage is not None else 0.0
            params["goodwill_penalty"] = goodwill if goodwill is not None else 0.0

        return Stage3ExtractedInputs(
            parameters=params,
            defaults_applied=defaults,
            missing_critical=missing,
            notes=notes,
        )

    # -------------------------------------------------------------------------
    # STAGE 4: Select Python Tool
    # -------------------------------------------------------------------------

    def select_tool(self, model_type: ModelType, inputs: Stage3ExtractedInputs) -> Stage4ToolSelection:
        """Select Python calculation function and format arguments."""
        tool_name = TOOL_MAPPING.get(model_type, "unknown_tool")
        return Stage4ToolSelection(
            tool_function=tool_name,
            arguments=inputs.parameters,
        )

    # -------------------------------------------------------------------------
    # STAGE 5: Execute Python Calculation
    # -------------------------------------------------------------------------

    def calculate(self, tool_sel: Stage4ToolSelection, model_type: ModelType) -> Stage5CalculationResult:
        """Invoke pure Python numerical calculation."""
        args = tool_sel.arguments
        
        if model_type == ModelType.EOQ:
            res = calculate_eoq(**args)
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            res = calculate_quantity_discount(**args)
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            res = calculate_rop(**args)
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            res = calculate_periodic_review(**args)
        elif model_type == ModelType.NEWSVENDOR:
            res = calculate_newsvendor(**args)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return Stage5CalculationResult(
            result_object=res,
            result_dict=res.to_dict(),
        )

    # -------------------------------------------------------------------------
    # STAGE 6: Verification & Sanity Checks
    # -------------------------------------------------------------------------

    def verify(
        self,
        model_type: ModelType,
        calc_result: Stage5CalculationResult,
        inputs: Stage3ExtractedInputs,
    ) -> Stage6Verification:
        """Run business invariant validation and sanity checks on calculations."""
        res_obj = calc_result.result_object
        base_verif: CalculationVerification = getattr(res_obj, "verification", CalculationVerification(is_valid=True))

        checks_passed = list(base_verif.checks_passed)
        warnings = list(base_verif.warnings)
        errors = list(base_verif.errors)

        if model_type == ModelType.EOQ:
            r: EOQResult = res_obj
            if r.eoq > 0:
                checks_passed.append(f"EOQ ({r.eoq:.2f} units) produces strictly positive order quantity.")
            if r.order_cycle_days > 365:
                warnings.append(
                    f"Order cycle length is {r.order_cycle_days:.1f} days (> 1 year); consider holding/spoilage risks."
                )

        elif model_type == ModelType.QUANTITY_DISCOUNT:
            qd: QuantityDiscountResult = res_obj
            if qd.optimal_total_annual_cost <= qd.baseline_total_annual_cost:
                checks_passed.append(
                    f"Selected tier achieves strictly non-negative savings (${qd.annual_cost_savings:.2f}) over baseline."
                )

        elif model_type == ModelType.SAFETY_STOCK_ROP:
            rop_res: SafetyStockROPResult = res_obj
            if rop_res.reorder_point >= rop_res.lead_time_demand:
                checks_passed.append("Reorder point correctly covers expected lead time demand plus non-negative buffer.")

        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            fpr: FixedPeriodReviewResult = res_obj
            if fpr.order_up_to_level >= fpr.expected_demand_protection_period:
                checks_passed.append("Target base stock level covers full exposure window (R + L).")

        elif model_type == ModelType.NEWSVENDOR:
            nv: NewsvendorResult = res_obj
            if 0.0 < nv.critical_ratio < 1.0:
                checks_passed.append(f"Critical ratio {nv.critical_ratio:.4f} is within open interval (0, 1).")
            if nv.expected_profit > 0:
                checks_passed.append(f"Expected profit is positive (${nv.expected_profit:.2f}).")
            else:
                warnings.append("Expected profit is non-positive; evaluate if product margins justify commercialization.")

        is_valid = len(errors) == 0 and base_verif.is_valid

        return Stage6Verification(
            is_valid=is_valid,
            checks_passed=checks_passed,
            warnings=warnings,
            errors=errors,
        )

    # -------------------------------------------------------------------------
    # STAGE 7: Explain & Synthesize Report
    # -------------------------------------------------------------------------

    def explain(
        self,
        question: str,
        model_id: Stage2ModelIdentification,
        inputs: Stage3ExtractedInputs,
        calc_result: Optional[Stage5CalculationResult],
        verification: Stage6Verification,
    ) -> Stage7Explanation:
        """Generate human-readable, executive-ready supply chain explanation."""
        if not calc_result or not verification.is_valid:
            err_msg = "\n".join(f"- {e}" for e in verification.errors) or "Missing required input parameters."
            missing_msg = "\n".join(f"- {m}" for m in inputs.missing_critical)
            exec_summary = "Calculation halted due to missing parameters or invalid input data."
            report = f"""### Supply Chain Analysis Incomplete

**Reason:** Could not complete calculation.

**Missing Parameters:**
{missing_msg}

**Errors Encountered:**
{err_msg}

Please provide the missing numerical values to run the inventory model.
"""
            return Stage7Explanation(
                executive_summary=exec_summary,
                detailed_explanation="Calculation could not proceed.",
                actionable_recommendations=["Provide the missing parameter values."],
                markdown_report=report,
            )

        res = calc_result.result_object
        mt = model_id.model_type

        # 1. EOQ Explanation
        if mt == ModelType.EOQ:
            r: EOQResult = res
            exec_summary = (
                f"Order **{r.recommended_order_quantity:,} units** per batch. "
                f"This results in approximately **{r.orders_per_year:.1f} orders/year** "
                f"(every **{r.order_cycle_days:.1f} operating days**), yielding a total annual inventory "
                f"cost of **${r.total_annual_inventory_cost:,.2f}**."
            )
            recs = [
                f"Set standard replenishment lot size to {r.recommended_order_quantity:,} units.",
                f"Schedule replenishment roughly every {round(r.order_cycle_days)} business days.",
                "Review order batching with supplier to ensure compatibility with pallet or master carton sizes.",
            ]
            report = f"""### 📦 Inventory Optimization: Economic Order Quantity (EOQ)

#### Executive Summary
{exec_summary}

---

#### 📐 Mathematical Breakdown
- **Formula:** $EOQ = \\sqrt{{\\frac{{2 \\cdot D \\cdot S}}{{H}}}}$
- **Substituted Values:**
  - Annual Demand ($D$): `{r.demand:,.0f}` units/year
  - Ordering Cost ($S$): `${r.ordering_cost:,.2f}` per order
  - Holding Cost ($H$): `${r.holding_cost_per_unit:,.2f}` per unit/year
- **Calculation:** $EOQ = \\sqrt{{\\frac{{2 \\times {r.demand:,.0f} \\times {r.ordering_cost:,.2f}}}{{{r.holding_cost_per_unit:,.2f}}}}} = {r.eoq:,.2f} \\approx {r.recommended_order_quantity:,}$ units.

#### 📊 Cost & Operational Metrics
| Metric | Value |
| :--- | :--- |
| **Recommended Order Quantity** | **{r.recommended_order_quantity:,} units** |
| **Annual Order Frequency** | **{r.orders_per_year:.2f} orders/year** |
| **Order Cycle Length** | **{r.order_cycle_days:.1f} days** ({r.work_days_per_year} workdays/yr) |
| **Annual Ordering Cost** | **${r.annual_ordering_cost:,.2f}** |
| **Annual Holding Cost** | **${r.annual_holding_cost:,.2f}** |
| **Total Annual Inventory Cost (Hold + Order)** | **${r.total_annual_inventory_cost:,.2f}** |
| **Annual Purchase Cost** | **${r.annual_purchase_cost:,.2f}** |
| **Grand Total Annual Cost** | **${r.grand_total_annual_cost:,.2f}** |

> [!NOTE]
> **Optimality Invariant:** At the true mathematical EOQ, Annual Ordering Cost equals Annual Holding Cost. In your operation, ordering cost is **${r.annual_ordering_cost:,.2f}** and holding cost is **${r.annual_holding_cost:,.2f}**, confirming minimal total inventory cost.
"""

        # 2. Quantity Discount Explanation
        elif mt == ModelType.QUANTITY_DISCOUNT:
            qd: QuantityDiscountResult = res
            exec_summary = (
                f"Optimal order quantity is **{qd.recommended_order_quantity:,} units** "
                f"at **${qd.optimal_unit_price:,.2f}/unit** (Tier {qd.optimal_tier_index + 1}). "
                f"This generates **${qd.annual_cost_savings:,.2f}** in annual savings "
                f"({qd.percentage_savings:.2f}% reduction) compared to the un-discounted baseline."
            )
            recs = [
                f"Order {qd.recommended_order_quantity:,} units per batch to unlock Tier {qd.optimal_tier_index + 1} discount pricing.",
                f"Total annual procurement and inventory cost will be minimized at ${qd.optimal_total_annual_cost:,.2f}.",
                "Ensure sufficient warehouse space is available to hold the higher lot size without incurring external storage fees.",
            ]
            
            # Format tier table
            tier_rows = []
            for t in qd.tier_evaluations:
                max_str = f"{t.max_qty:,.0f}" if t.max_qty is not None else "∞"
                is_opt = "⭐ **Optimal**" if t.tier_index == qd.optimal_tier_index else ""
                tier_rows.append(
                    f"| Tier {t.tier_index + 1} | {t.min_qty:,.0f} - {max_str} | ${t.unit_price:,.2f} | "
                    f"{t.unconstrained_eoq:,.1f} | **{t.feasible_order_quantity:,.0f}** | "
                    f"${t.annual_purchase_cost:,.2f} | ${t.annual_ordering_cost:,.2f} | ${t.annual_holding_cost:,.2f} | "
                    f"**${t.total_annual_cost:,.2f}** | {is_opt} |"
                )
            tier_table = "\n".join(tier_rows)

            report = f"""### 🏷️ Inventory Optimization: Quantity Discount Analysis

#### Executive Summary
{exec_summary}

---

#### 📋 Price Break Evaluation Table
| Tier | Order Range | Unit Price | Unconstrained EOQ | Feasible Order Qty | Purchase Cost | Ordering Cost | Holding Cost | Total Annual Cost | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{tier_table}

#### 💡 Analytical Insight
The algorithm checks each price bracket:
1. Calculates unconstrained EOQ for that bracket's holding cost rate.
2. Checks whether unconstrained EOQ falls inside the bracket. If it is below the tier minimum, it rounds up to the exact price break point (`{qd.optimal_order_quantity:,.0f}` units).
3. Evaluates total cost (Purchase + Ordering + Holding) across all tiers.
4. **Tier {qd.optimal_tier_index + 1}** produces the lowest grand total cost of **${qd.optimal_total_annual_cost:,.2f}**.
"""

        # 3. Safety Stock & ROP Explanation
        elif mt == ModelType.SAFETY_STOCK_ROP:
            ss_res: SafetyStockROPResult = res
            exec_summary = (
                f"Place a replenishment order whenever inventory drops to **{ss_res.recommended_reorder_point:,} units** (Reorder Point). "
                f"This includes a safety buffer of **{ss_res.recommended_safety_stock:,} units** "
                f"to protect against demand spikes and lead-time delays at a **{ss_res.service_level:.1%} Cycle Service Level**."
            )
            recs = [
                f"Set ERP/WMS Reorder Point parameter to {ss_res.recommended_reorder_point:,} units.",
                f"Maintain a safety stock buffer of {ss_res.recommended_safety_stock:,} units.",
                f"Expected stockout risk during replenishment lead time is {ss_res.stockout_risk_per_cycle:.1%}.",
            ]

            tradeoff_rows = []
            for row in ss_res.service_level_tradeoffs:
                is_curr = "👈 **Current Target**" if abs(row["service_level"] - ss_res.service_level) < 1e-3 else ""
                tradeoff_rows.append(
                    f"| {row['service_level']:.1%} | {row['z_score']:.3f} | {row['safety_stock']:,.1f} | "
                    f"**{row['reorder_point']:,.1f}** | ${row['annual_ss_holding_cost']:,.2f} | {row['stockout_risk_pct']:.1f}% | {is_curr} |"
                )
            tradeoff_table = "\n".join(tradeoff_rows)

            report = f"""### 🛡️ Inventory Optimization: Safety Stock & Reorder Point (ROP)

#### Executive Summary
{exec_summary}

---

#### 📐 Mathematical Breakdown
- **Variability Scenario:** `{ss_res.variability_case}`
- **Expected Demand During Lead Time (DDLT):** $\\bar{{d}} \\times L = {ss_res.avg_daily_demand:,.2f} \\times {ss_res.lead_time_days:.1f} = {ss_res.lead_time_demand:,.2f}$ units
- **Combined Lead Time Standard Deviation ($\\sigma_L$):** `{ss_res.combined_std_dev:,.2f}` units
- **Z-Score for {ss_res.service_level:.1%} Service Level:** `{ss_res.z_score:.3f}`
- **Safety Stock ($SS$):** $z \\times \\sigma_L = {ss_res.z_score:.3f} \\times {ss_res.combined_std_dev:,.2f} = {ss_res.safety_stock:,.2f} \\approx$ **{ss_res.recommended_safety_stock:,} units**
- **Reorder Point ($ROP$):** $DDLT + SS = {ss_res.lead_time_demand:,.2f} + {ss_res.safety_stock:,.2f} = {ss_res.reorder_point:,.2f} \\approx$ **{ss_res.recommended_reorder_point:,} units**

#### 📈 Service Level Trade-Off Table
| Service Level | Z-Score | Safety Stock | Reorder Point | Annual SS Holding Cost | Stockout Risk | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{tradeoff_table}
"""

        # 4. Fixed-Period Review Explanation
        elif mt == ModelType.FIXED_PERIOD_REVIEW:
            fpr: FixedPeriodReviewResult = res
            exec_summary = (
                f"Target Order-Up-To Level is **{fpr.recommended_target_level:,} units**. "
                f"Given current on-hand ({fpr.current_on_hand:,.0f}) and on-order ({fpr.inventory_on_order:,.0f}), "
                f"your current inventory position is **{fpr.inventory_position:,.0f} units**. "
                f"You should place an order for **{fpr.recommended_order_quantity:,.0f} units** today."
            )
            recs = [
                f"Issue a purchase order for {int(round(fpr.recommended_order_quantity)):,} units immediately.",
                f"Next inventory inspection scheduled in {fpr.review_period_days:.0f} days.",
                f"Note: Periodic review requires {fpr.periodic_safety_stock_multiplier:.2f}x higher safety stock than continuous review due to the longer exposure window (R + L = {fpr.protection_interval_days:.1f} days).",
            ]
            report = f"""### 📅 Inventory Optimization: Fixed-Period Review Policy (R, S)

#### Executive Summary
{exec_summary}

---

#### 📐 Mathematical Breakdown
- **Review Period ($R$):** `{fpr.review_period_days:.1f}` days
- **Supplier Lead Time ($L$):** `{fpr.lead_time_days:.1f}` days
- **Total Protection Interval ($R + L$):** `{fpr.protection_interval_days:.1f}` days
- **Expected Demand over ($R + L$):** $\\bar{{d}} \\times (R + L) = {fpr.avg_daily_demand:,.2f} \\times {fpr.protection_interval_days:.1f} = {fpr.expected_demand_protection_period:,.2f}$ units
- **Std Dev over ($R + L$):** $\\sigma_d \\times \\sqrt{{R + L}} = {fpr.std_daily_demand:,.2f} \\times \\sqrt{{{fpr.protection_interval_days:.1f}}} = {fpr.std_dev_protection_period:,.2f}$ units
- **Periodic Safety Stock ($SS$):** $z \\times \\sigma_{{R+L}} = {fpr.z_score:.3f} \\times {fpr.std_dev_protection_period:,.2f} = {fpr.safety_stock:,.2f} \\approx$ **{fpr.recommended_safety_stock:,} units**
- **Target Stock Level ($S$):** **{fpr.order_up_to_level:,.2f} units** (Round to **{fpr.recommended_target_level:,}**)

#### 📊 Current Inventory State & Order Calculation
| Metric | Value |
| :--- | :--- |
| **Current On-Hand Inventory** | **{fpr.current_on_hand:,.0f} units** |
| **Inventory On-Order (Pipeline)** | **{fpr.inventory_on_order:,.0f} units** |
| **Backorders / Backlog** | **{fpr.backorders:,.0f} units** |
| **Current Effective Inventory Position** | **{fpr.inventory_position:,.0f} units** |
| **Target Stock Level ($S$)** | **{fpr.recommended_target_level:,} units** |
| **Recommended Order Quantity ($Q = \\max(0, S - IP)$)** | **{fpr.recommended_order_quantity:,.0f} units** |
"""

        # 5. Newsvendor Explanation
        elif mt == ModelType.NEWSVENDOR:
            nv: NewsvendorResult = res
            exec_summary = (
                f"Optimal order quantity is **{nv.recommended_order_quantity:,} units** "
                f"based on a Critical Ratio of **{nv.critical_ratio:.1%}**. "
                f"Expected profit is **${nv.expected_profit:,.2f}** with an expected in-stock service level of **{nv.in_stock_probability:.1%}**."
            )
            recs = [
                f"Procure {nv.recommended_order_quantity:,} units for this single sales cycle.",
                f"Budget for ${nv.expected_purchase_cost:,.2f} in procurement expenditure.",
                f"Expected leftover inventory is {nv.expected_leftover_units:,.1f} units, recoverable for ${nv.expected_salvage_revenue:,.2f} in salvage value.",
            ]
            report = f"""### 📰 Single-Period Stochastic Model: Newsvendor

#### Executive Summary
{exec_summary}

---

#### 📐 Marginal Economic Breakdown
- **Selling Price ($P$):** `${nv.selling_price:,.2f}`
- **Unit Cost ($C$):** `${nv.unit_cost:,.2f}`
- **Salvage Value ($S_v$):** `${nv.salvage_value:,.2f}`
- **Cost of Underage ($C_u$):** Profit lost per unserved customer = $P - C = {nv.selling_price:,.2f} - {nv.unit_cost:,.2f} = ${nv.cost_underage:,.2f}$
- **Cost of Overage ($C_o$):** Loss per unsold unit = $C - S_v = {nv.unit_cost:,.2f} - {nv.salvage_value:,.2f} = ${nv.cost_overage:,.2f}$
- **Critical Ratio ($CR^*$):** $\\frac{{C_u}}{{C_u + C_o}} = \\frac{{{nv.cost_underage:,.2f}}}{{{nv.cost_underage:,.2f} + {nv.cost_overage:,.2f}}} =$ **{nv.critical_ratio:.4f}** ({nv.critical_ratio:.2%})
- **Z-Score ($z^*$):** `{nv.z_score:.3f}`
- **Optimal Order Quantity ($Q^*$):** $\\mu + z^* \\cdot \\sigma = {nv.mean_demand:,.0f} + ({nv.z_score:.3f} \\times {nv.std_demand:,.0f}) = {nv.optimal_order_quantity:,.2f} \\approx$ **{nv.recommended_order_quantity:,} units**

#### 💰 Expected Financial Performance
| Metric | Value |
| :--- | :--- |
| **Optimal Stocking Quantity** | **{nv.recommended_order_quantity:,} units** |
| **Expected Sales** | **{nv.expected_sales:,.1f} units** |
| **Expected Leftover Inventory** | **{nv.expected_leftover_units:,.1f} units** |
| **Expected Lost Sales (Stockouts)** | **{nv.expected_lost_sales:,.1f} units** |
| **Expected Revenue (Sales + Salvage)** | **${nv.expected_revenue:,.2f}** |
| **Expected Procurement Cost** | **${nv.expected_purchase_cost:,.2f}** |
| **Expected Net Profit** | **${nv.expected_profit:,.2f}** |
| **In-Stock Service Probability** | **{nv.in_stock_probability:.1%}** |
"""
        else:
            exec_summary = "Unrecognized model."
            recs = []
            report = "Model could not be explained."

        return Stage7Explanation(
            executive_summary=exec_summary,
            detailed_explanation=report,
            actionable_recommendations=recs,
            markdown_report=report,
        )

    # -------------------------------------------------------------------------
    # MAIN AGENT ENTRYPOINT: Complete Pipeline Execution
    # -------------------------------------------------------------------------

    def answer_question(self, question: str, override_model: Optional[ModelType] = None) -> AgentPipelineTrace:
        """Run the full 7-stage Supply Chain Decision Agent pipeline."""
        # Stage 1: Question
        raw_question = question.strip()

        # Stage 2: Identify Model
        if override_model and override_model != ModelType.UNKNOWN:
            model_id = Stage2ModelIdentification(
                model_type=override_model,
                model_name=MODEL_DISPLAY_NAMES[override_model],
                confidence=1.0,
                rationale="User manually selected this model override.",
            )
        else:
            model_id = self.identify_model(raw_question)

        # Stage 3: Extract Inputs
        inputs = self.extract_inputs(raw_question, model_id.model_type)

        # Stage 4: Select Python Tool
        tool_sel = self.select_tool(model_id.model_type, inputs)

        # Stage 5: Calculate (if all critical inputs available)
        calc_result: Optional[Stage5CalculationResult] = None
        if not inputs.missing_critical and model_id.model_type != ModelType.UNKNOWN:
            calc_result = self.calculate(tool_sel, model_id.model_type)

        # Stage 6: Verify
        if calc_result:
            verification = self.verify(model_id.model_type, calc_result, inputs)
        else:
            verification = Stage6Verification(
                is_valid=False,
                checks_passed=[],
                warnings=[],
                errors=[f"Cannot calculate: missing required inputs: {', '.join(inputs.missing_critical)}"],
            )

        # Stage 7: Explain
        explanation = self.explain(raw_question, model_id, inputs, calc_result, verification)

        return AgentPipelineTrace(
            question=raw_question,
            model_identification=model_id,
            extracted_inputs=inputs,
            tool_selection=tool_sel,
            calculation=calc_result,
            verification=verification,
            explanation=explanation,
        )
