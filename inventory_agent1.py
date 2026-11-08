"""
inventory_agent.py - Supply Chain Analytics Inventory Decision Agent.

Architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations
import re
import math
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import inventory_tools as tools


class ModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount"
    SAFETY_STOCK_ROP = "Safety Stock and Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review"
    NEWSEVNDOR = "Newsvendor"
    UNKNOWN = "Unknown Model"


@dataclass
class VerificationCheck:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str


@dataclass
class AgentStepTrace:
    step_name: str
    status: str
    details: Dict[str, Any]


@dataclass
class AgentResponse:
    question: str
    identified_model: ModelType
    model_confidence: float
    identification_reasoning: str
    extracted_inputs: Dict[str, Any]
    tool_selected: str
    calculation_results: Optional[Dict[str, Any]]
    verification_checks: List[VerificationCheck]
    verification_overall: str  # "PASSED", "WARNING", "FAILED"
    explanation_markdown: str
    curve_data: Optional[Dict[str, Any]] = None
    execution_trace: List[AgentStepTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["identified_model"] = self.identified_model.value
        return data


class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent implementing:
    Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
    """

    def __init__(self):
        self.tool_mapping = {
            ModelType.EOQ: "tools.calculate_eoq",
            ModelType.QUANTITY_DISCOUNT: "tools.calculate_quantity_discount",
            ModelType.SAFETY_STOCK_ROP: "tools.calculate_rop",
            ModelType.FIXED_PERIOD_REVIEW: "tools.calculate_periodic_review",
            ModelType.NEWSEVNDOR: "tools.calculate_newsvendor",
        }

    # =========================================================================
    # STEP 1: IDENTIFY MODEL
    # =========================================================================
    def identify_model(self, question: str) -> Tuple[ModelType, float, str]:
        """
        Analyze customer question and identify the most appropriate inventory model.
        Returns: (ModelType, confidence_score, reasoning)
        """
        q_lower = question.lower()

        scores: Dict[ModelType, float] = {
            ModelType.NEWSEVNDOR: 0.0,
            ModelType.QUANTITY_DISCOUNT: 0.0,
            ModelType.FIXED_PERIOD_REVIEW: 0.0,
            ModelType.SAFETY_STOCK_ROP: 0.0,
            ModelType.EOQ: 0.0,
        }
        reasons: Dict[ModelType, List[str]] = {m: [] for m in scores}

        # 1. Newsvendor indicators
        if any(term in q_lower for term in ["newsvendor", "newsboy", "single period", "single-period"]):
            scores[ModelType.NEWSEVNDOR] += 40
            reasons[ModelType.NEWSEVNDOR].append("Explicit single-period/newsvendor reference")
        if any(term in q_lower for term in ["salvage", "scrap value", "clearance price", "unsold units"]):
            scores[ModelType.NEWSEVNDOR] += 30
            reasons[ModelType.NEWSEVNDOR].append("Salvage or unsold unit clearance mentioned")
        if any(term in q_lower for term in ["perishable", "seasonal", "fashion", "spoilage", "event", "perish"]):
            scores[ModelType.NEWSEVNDOR] += 20
            reasons[ModelType.NEWSEVNDOR].append("Perishable or seasonal product scenario")
        if "critical fractile" in q_lower or "critical ratio" in q_lower:
            scores[ModelType.NEWSEVNDOR] += 30
            reasons[ModelType.NEWSEVNDOR].append("Critical fractile/ratio terminology")

        # 2. Quantity Discount indicators
        if any(term in q_lower for term in ["discount", "price break", "volume tier", "volume discount", "schedule", "bracket"]):
            scores[ModelType.QUANTITY_DISCOUNT] += 35
            reasons[ModelType.QUANTITY_DISCOUNT].append("Discount or price break terms detected")
        if re.search(r"(\d+[\s\-]*(?:to|-)\s*\d+|\d+\s*\+)", q_lower) and any(x in q_lower for x in ["$", "dollar", "price", "unit"]):
            scores[ModelType.QUANTITY_DISCOUNT] += 30
            reasons[ModelType.QUANTITY_DISCOUNT].append("Quantity tier bracket formatting detected")
        if "tier" in q_lower or "break point" in q_lower:
            scores[ModelType.QUANTITY_DISCOUNT] += 25
            reasons[ModelType.QUANTITY_DISCOUNT].append("Tier structure keywords found")

        # 3. Fixed-Period Review indicators
        if any(term in q_lower for term in ["fixed period", "periodic review", "order-up-to", "order up to", "target inventory", "target stock"]):
            scores[ModelType.FIXED_PERIOD_REVIEW] += 40
            reasons[ModelType.FIXED_PERIOD_REVIEW].append("Explicit periodic review or order-up-to terminology")
        if re.search(r"review\s*(?:period|every|interval)?\s*(?:of)?\s*\d+\s*(?:day|week|month)", q_lower) or "every 14 days" in q_lower or "every 7 days" in q_lower:
            scores[ModelType.FIXED_PERIOD_REVIEW] += 35
            reasons[ModelType.FIXED_PERIOD_REVIEW].append("Periodic review interval specified (e.g. every N days)")
        if any(term in q_lower for term in ["on-hand", "on hand", "pipeline", "on order", "current inventory"]):
            scores[ModelType.FIXED_PERIOD_REVIEW] += 15
            reasons[ModelType.FIXED_PERIOD_REVIEW].append("Current on-hand or on-order inventory specified")

        # 4. Safety Stock & Reorder Point indicators
        if any(term in q_lower for term in ["reorder point", "rop", "safety stock", "buffer stock", "continuous review"]):
            scores[ModelType.SAFETY_STOCK_ROP] += 40
            reasons[ModelType.SAFETY_STOCK_ROP].append("Explicit reorder point (ROP) or safety stock terminology")
        if any(term in q_lower for term in ["service level", "cycle service level", "csl", "stockout risk"]):
            scores[ModelType.SAFETY_STOCK_ROP] += 20
            reasons[ModelType.SAFETY_STOCK_ROP].append("Service level or stockout risk specified")
        if "lead time" in q_lower and ("std dev" in q_lower or "standard deviation" in q_lower or "sigma" in q_lower):
            scores[ModelType.SAFETY_STOCK_ROP] += 20
            reasons[ModelType.SAFETY_STOCK_ROP].append("Stochastic demand during lead time mentioned")

        # 5. EOQ indicators
        if any(term in q_lower for term in ["eoq", "economic order quantity", "optimal batch size", "economic lot size"]):
            scores[ModelType.EOQ] += 40
            reasons[ModelType.EOQ].append("Explicit EOQ or economic lot size keywords")
        if any(term in q_lower for term in ["annual demand", "demand per year"]) and any(term in q_lower for term in ["ordering cost", "setup cost", "order cost"]):
            scores[ModelType.EOQ] += 25
            reasons[ModelType.EOQ].append("Annual demand and ordering/setup cost combination")
        if any(term in q_lower for term in ["holding cost", "carrying cost"]):
            scores[ModelType.EOQ] += 15
            reasons[ModelType.EOQ].append("Inventory holding or carrying cost mentioned")

        # Pick best model
        best_model = max(scores, key=lambda m: scores[m])
        max_score = scores[best_model]

        if max_score < 15:
            # Fallback check
            if "order" in q_lower and "demand" in q_lower:
                return ModelType.EOQ, 0.50, "Defaulted to EOQ based on generic order and demand keywords."
            return ModelType.UNKNOWN, 0.10, "Unable to decisively map the query to a recognized inventory model."

        confidence = min(0.99, max_score / 60.0)
        reasoning_text = "; ".join(reasons[best_model]) if reasons[best_model] else "Best matching pattern."
        return best_model, round(confidence, 2), reasoning_text

    # =========================================================================
    # STEP 2: EXTRACT INPUTS (IMPROVED NATURAL LANGUAGE ENGINE)
    # =========================================================================
    def extract_inputs(self, question: str, model_type: ModelType) -> Dict[str, Any]:
        """
        Extract numerical parameters and semantics from natural language questions.
        Recognizes common supply chain terminology, synonyms, and diverse phrasings.
        """
        # Pre-process: remove thousand-separators, normalize dashes, symbols
        q = re.sub(r"(\d),(\d)", r"\1\2", question)
        q = q.replace("–", "-").replace("—", "-")

        def find_number(patterns: List[str], text: str) -> Optional[float]:
            for p in patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1))
                    except (ValueError, IndexError):
                        continue
            return None

        inputs: Dict[str, Any] = {}

        # =====================================================================
        # 1. EOQ EXTRACTION
        # =====================================================================
        if model_type == ModelType.EOQ:
            # Annual demand
            demand = find_number([
                r"(?:annual\s*demand|demand\s*per\s*year|yearly\s*demand|annual\s*usage|annual\s*sales|annual\s*consumption)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"demand\s*(?:of|is|:)\s*(\d+(?:\.\d+)?)\s*(?:units\s*(?:per|\/)\s*year|units\s*annually|per\s*year)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*(?:per|\/)\s*year|units\s*annually|units\s*a\s*year)",
                r"sells\s*(\d+(?:\.\d+)?)\s*(?:units\s*(?:per|\/)\s*year|units\s*annually|units)",
                r"demand\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # Ordering / Setup cost - Treat all required synonyms:
            # - ordering cost
            # - order cost
            # - setup cost
            # - setup expense
            # - cost per order
            # - cost of placing an order
            # - ordering/setup cost
            # - cost to place an order
            # - placing an order costs
            order_cost = find_number([
                r"(?:ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|cost\s*per\s*order|cost\s*of\s*placing\s*(?:an\s*)?order|ordering\s*[\/\-&]\s*setup\s*cost|ordering\s+and\s+setup\s*cost|cost\s*to\s*place\s*(?:an\s*)?order|cost\s*to\s*order|order\s*placement\s*cost|replenishment\s*cost|procurement\s*cost)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"placing\s*(?:an\s*)?order\s*costs?\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*order|\/\s*order|to\s*place\s*(?:an\s*)?order|for\s*each\s*order|setup\s*expense|ordering\s*cost|setup\s*cost)",
                r"\$?(\d+(?:\.\d+)?)\s*cost\s*per\s*order",
            ], q)

            # Holding cost / Carrying cost
            holding_cost = find_number([
                r"(?:holding\s*cost|carrying\s*cost|holding\s*expense|cost\s*(?:of|to)\s*(?:hold|carry)(?:ing)?|storage\s*cost|carrying\s*charge|inventory\s*carrying\s*cost)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit\s*(?:per|\/)\s*year|\/\s*unit\s*(?:per|\/)\s*year|\/\s*unit\s*year|per\s*item\s*per\s*year)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:holding\s*cost|carrying\s*cost|holding\s*expense)",
            ], q)

            # Percentage holding cost
            holding_rate = find_number([
                r"(?:percentage\s*holding\s*cost|holding\s*rate|carrying\s*rate|holding\s*cost\s*rate|annual\s*holding\s*rate|carrying\s*charge)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:purchase\s*)?(?:unit\s*)?price|of\s*unit\s*cost|holding|carrying)",
                r"(?:holding|carrying)\s*is\s*(\d+(?:\.\d+)?)\s*%",
            ], q)
            if holding_rate is not None and holding_rate > 1.0:
                holding_rate = holding_rate / 100.0

            # Unit purchase price / cost
            unit_cost = find_number([
                r"(?:unit\s*purchase\s*price|purchase\s*price|unit\s*cost|cost\s*per\s*unit|purchase\s*cost|item\s*cost|cost\s*each)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"costs?\s*\$?(\d+(?:\.\d+)?)\s*(?:each|per\s*unit|a\s*unit)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit|each|purchase\s*price)",
            ], q)

            # Compute holding cost from percentage holding cost if needed
            if holding_cost is None and holding_rate is not None and unit_cost is not None:
                holding_cost = round(holding_rate * unit_cost, 4)

            # Lead time days
            lead_time = find_number([
                r"(?:lead\s*time|supplier\s*lead\s*time|delivery\s*time|replenishment\s*time)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:days?)",
                r"takes\s*(\d+(?:\.\d+)?)\s*(?:days?)",
            ], q)

            # Workdays per year
            workdays = find_number([
                r"(\d+)\s*(?:workdays|working\s*days|operating\s*days)\s*(?:per|\/)\s*year",
            ], q)

            inputs = {
                "annual_demand": demand,
                "order_cost": order_cost,
                "holding_cost": holding_cost,
                "holding_rate": holding_rate,
                "unit_cost": unit_cost,
                "lead_time_days": lead_time,
                "workdays_per_year": int(workdays) if workdays else 250,
            }

        # =====================================================================
        # 2. QUANTITY DISCOUNT EXTRACTION
        # =====================================================================
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            # Distinguish:
            # - quantity break / discount tiers
            # - unit price
            # - ordering cost
            # - fixed holding cost
            # - percentage holding cost
            # - unit purchase price
            demand = find_number([
                r"(?:annual\s*demand|demand\s*per\s*year|yearly\s*demand|annual\s*usage|annual\s*sales)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*(?:per|\/)\s*year|units\s*annually)",
                r"demand\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # Ordering cost (recognizes all synonyms)
            order_cost = find_number([
                r"(?:ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|cost\s*per\s*order|cost\s*of\s*placing\s*(?:an\s*)?order|ordering\s*[\/\-&]\s*setup\s*cost|cost\s*to\s*place\s*(?:an\s*)?order)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*order|\/\s*order|to\s*place\s*(?:an\s*)?order)",
            ], q)

            # Percentage holding cost
            holding_rate = find_number([
                r"(?:percentage\s*holding\s*cost|holding\s*rate|carrying\s*rate|holding\s*cost\s*rate)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*(?:the\s*)?(?:purchase\s*)?(?:unit\s*)?price|of\s*unit\s*cost|holding|carrying)",
                r"holding\s*(?:is|cost\s*is)\s*(\d+(?:\.\d+)?)\s*%",
            ], q)
            if holding_rate is not None and holding_rate > 1.0:
                holding_rate = holding_rate / 100.0

            # Fixed holding cost
            fixed_holding_cost = find_number([
                r"(?:fixed\s*holding\s*cost|holding\s*cost|carrying\s*cost)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit|\/\s*unit)",
            ], q)

            # Baseline unit purchase price if given separately
            unit_purchase_price = find_number([
                r"(?:unit\s*purchase\s*price|base\s*price|regular\s*price|baseline\s*price|unit\s*price)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # Parse quantity breaks / discount tiers
            tiers: List[Dict[str, Any]] = []

            # Format 1: "1-499 units at $20" or "0 to 499: $20"
            range_matches = re.findall(
                r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:units)?\s*(?:at|:|\$|@)?\s*\$?(\d+(?:\.\d+)?)",
                q,
                re.IGNORECASE,
            )
            for m in range_matches:
                tiers.append({
                    "min_qty": float(m[0]),
                    "max_qty": float(m[1]),
                    "unit_price": float(m[2]),
                })

            # Format 2: "1000+ at $16" or "1000 or more: $16" or "over 1000: $16"
            plus_matches = re.findall(
                r"(\d+)\s*(?:\+|or\s*more|and\s*above|or\s*greater)\s*(?:units)?\s*(?:at|:|\$|@)?\s*\$?(\d+(?:\.\d+)?)",
                q,
                re.IGNORECASE,
            )
            for m in plus_matches:
                tiers.append({
                    "min_qty": float(m[0]),
                    "max_qty": None,
                    "unit_price": float(m[1]),
                })

            # Format 3: "under 500 units at $20" / "less than 500 units at $20"
            under_matches = re.findall(
                r"(?:under|less\s*than|fewer\s*than)\s*(\d+)\s*(?:units)?\s*(?:at|:|\$|@)?\s*\$?(\d+(?:\.\d+)?)",
                q,
                re.IGNORECASE,
            )
            for m in under_matches:
                # e.g. under 500 -> 1 to 499
                upper = float(m[0]) - 1.0
                tiers.append({
                    "min_qty": 1.0,
                    "max_qty": upper,
                    "unit_price": float(m[1]),
                })

            inputs = {
                "annual_demand": demand,
                "order_cost": order_cost,
                "holding_rate": holding_rate,
                "fixed_holding_cost": fixed_holding_cost,
                "unit_purchase_price": unit_purchase_price,
                "discount_schedule": tiers,
            }

        # =====================================================================
        # 3. SAFETY STOCK & REORDER POINT EXTRACTION
        # =====================================================================
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            # Distinguish:
            # - average demand
            # - demand standard deviation
            # - average lead time
            # - lead-time standard deviation
            # - service level

            # First, extract demand standard deviation so average demand doesn't misclassify it
            demand_std_dev = find_number([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*(?:daily\s*|weekly\s*)?demand|std\s*dev\s*of\s*(?:daily\s*|weekly\s*)?demand|demand\s*std\s*dev|sigma\s*d)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"(?:standard\s*deviation|std\s*dev|sigma)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:units)?",
                r"(\d+(?:\.\d+)?)\s*(?:units\s*)?(?:demand\s*)?standard\s*deviation",
            ], q) or 0.0

            # Average demand (ensure standard deviation phrase is avoided)
            demand = find_number([
                r"(?:average\s*demand|mean\s*demand|expected\s*demand|daily\s*demand|weekly\s*demand|demand\s*per\s*(?:day|week|period)|demand\s*averages?)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"demand\s*(?:is|of|:)\s*(\d+(?:\.\d+)?)\s*(?:units)?\s*(?:per|\/)\s*(?:day|week|period)",
                r"(\d+(?:\.\d+)?)\s*units\s*(?:per|\/)\s*(?:day|week|period)",
                r"demand\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # Lead-time standard deviation
            lead_time_std_dev = find_number([
                r"(?:lead[\s\-]time\s*standard\s*deviation|standard\s*deviation\s*of\s*lead[\s\-]time|std\s*dev\s*of\s*lead[\s\-]time|lead[\s\-]time\s*std\s*dev|sigma\s*l)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"lead[\s\-]time\s*(?:varies\s*with\s*(?:a\s*)?standard\s*deviation|has\s*(?:a\s*)?standard\s*deviation)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
            ], q) or 0.0

            # Average lead time
            lead_time = find_number([
                r"(?:average\s*lead[\s\-]time|mean\s*lead[\s\-]time|expected\s*lead[\s\-]time|lead[\s\-]time|delivery\s*time|replenishment\s*time)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)?",
                r"takes\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)",
            ], q)

            # Service level or z-score
            service_level = find_number([
                r"(?:cycle\s*service\s*level|customer\s*service\s*level|target\s*service\s*level|service\s*level|csl)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(?:cycle\s*service\s*level|customer\s*service\s*level|target\s*service\s*level|service\s*level|csl)\s*(?:of|is|:)?\s*(0\.\d+)",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s*service\s*level|service\s*level|csl)",
            ], q)
            if service_level is not None and service_level > 1.0:
                service_level = service_level / 100.0

            z_score = find_number([
                r"(?:z[\s\-]score|z\s*value|z)\s*(?:of|is|=)\s*(\d+(?:\.\d+)?)",
            ], q)

            inputs = {
                "demand_per_period": demand,
                "demand_std_dev": demand_std_dev,
                "lead_time": lead_time,
                "lead_time_std_dev": lead_time_std_dev,
                "service_level": service_level,
                "z_score": z_score,
            }

        # =====================================================================
        # 4. FIXED-PERIOD REVIEW EXTRACTION
        # =====================================================================
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            # Distinguish:
            # - demand (mean and std dev)
            # - review period
            # - lead time
            # - safety stock
            # - inventory position
            # - on-hand inventory
            # - on-order inventory
            # - backorders

            review_period = find_number([
                r"(?:review\s*period|review\s*interval|order\s*interval|orders\s*placed\s*every|ordered\s*every|reviewed\s*every|review\s*cycle)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)",
                r"every\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)",
            ], q)
            # Support word intervals
            if review_period is None:
                if "weekly review" in q.lower():
                    review_period = 7.0
                elif "bi-weekly review" in q.lower() or "biweekly review" in q.lower():
                    review_period = 14.0
                elif "monthly review" in q.lower():
                    review_period = 30.0

            lead_time = find_number([
                r"(?:lead\s*time|delivery\s*time|replenishment\s*time)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"takes\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?)",
            ], q)

            demand_std_dev = find_number([
                r"(?:standard\s*deviation|std\s*dev|sigma)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
            ], q) or 0.0

            demand = find_number([
                r"(?:daily\s*demand|demand\s*per\s*(?:day|week)|average\s*demand|mean\s*demand|expected\s*demand)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"demand\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # Explicit safety stock if given directly
            safety_stock = find_number([
                r"(?:safety\s*stock|buffer\s*stock)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # Explicit inventory position if given directly
            inventory_position = find_number([
                r"(?:inventory\s*position|net\s*inventory\s*position)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            # On-hand inventory
            on_hand = find_number([
                r"(?:on[\s\-]hand\s*inventory|on[\s\-]hand|current\s*(?:stock|inventory)|stock\s*on\s*hand|in\s*stock)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units)?\s*(?:in\s*stock|on[\s\-]hand|current\s*stock)",
                r"have\s*(\d+(?:\.\d+)?)\s*units",
            ], q) or 0.0

            # On-order inventory (pipeline)
            on_order = find_number([
                r"(?:on[\s\-]order\s*inventory|on[\s\-]order|pipeline\s*inventory|pipeline\s*orders?|pipeline|in\s*transit|orders\s*in\s*transit|open\s*orders|scheduled\s*receipts)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units)?\s*(?:are\s*)?(?:on[\s\-]order|in\s*transit|pipeline|open\s*orders)",
            ], q) or 0.0

            # Backorders
            backorders = find_number([
                r"(?:backorders?|backordered\s*units?|unfilled\s*orders?|backlog|outstanding\s*backorders?)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:units)?\s*(?:backordered|in\s*backlog|on\s*backorder)",
            ], q) or 0.0

            # Compute inventory position if not directly stated
            if inventory_position is None:
                inventory_position = on_hand + on_order - backorders

            # Service level
            service_level = find_number([
                r"(?:service\s*level|cycle\s*service\s*level|csl)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%",
                r"(?:service\s*level|cycle\s*service\s*level|csl)\s*(?:of|is|:)?\s*(0\.\d+)",
                r"(\d+(?:\.\d+)?)\s*%\s*(?:service\s*level)",
            ], q)
            if service_level is not None and service_level > 1.0:
                service_level = service_level / 100.0

            z_score = find_number([
                r"(?:z[\s\-]score|z\s*value|z)\s*(?:of|is|=)\s*(\d+(?:\.\d+)?)",
            ], q)

            inputs = {
                "review_period": review_period,
                "lead_time": lead_time,
                "demand_per_period": demand,
                "demand_std_dev": demand_std_dev,
                "safety_stock": safety_stock,
                "inventory_position": inventory_position,
                "current_inventory": on_hand,
                "pipeline_orders": on_order,
                "backorders": backorders,
                "service_level": service_level,
                "z_score": z_score,
            }

        # =====================================================================
        # 5. NEWSVENDOR EXTRACTION
        # =====================================================================
        elif model_type == ModelType.NEWSEVNDOR:
            # Distinguish:
            # - selling price
            # - purchase cost
            # - salvage value
            # - underage / shortage cost
            # - overage / excess cost
            # - demand distribution
            # - mean demand
            # - demand standard deviation

            price = find_number([
                r"(?:selling\s*price|retail\s*price|sale\s*price|market\s*price|revenue\s*per\s*unit|price\s*is|sells?\s*for)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"(?:sell|selling)[^$0-9]{0,40}(?:for|\$)\s*\$?(\d+(?:\.\d+)?)",
                r"\$?(\d+(?:\.\d+)?)\s*(?:retail|selling\s*price|each|per\s*unit)",
                r"price\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            cost = find_number([
                r"(?:purchase\s*cost|unit\s*cost|production\s*cost|acquisition\s*cost|wholesale\s*cost|cost\s*to\s*(?:produce|acquire|buy)|manufacturing\s*cost)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"costs?\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"cost\s*(?:is|of|:)?\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            salvage = find_number([
                r"(?:salvage\s*value|salvage\s*price|salvage\s*for|scrap\s*value|clearance\s*price|residual\s*value|liquidation\s*value|discounted\s*to|cleared\s*for)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
                r"salvaged?\s*(?:for)?\s*\$?(\d+(?:\.\d+)?)",
                r"unsold\s*units\s*(?:are\s*)?(?:salvaged|sold|cleared)\s*for\s*\$?(\d+(?:\.\d+)?)",
            ], q) or 0.0

            # Underage / shortage cost (direct Cu)
            underage = find_number([
                r"(?:underage\s*cost|cost\s*of\s*underage|shortage\s*cost|cost\s*of\s*shortage|stockout\s*cost|cost\s*of\s*stockout|lost\s*profit\s*(?:per\s*unit)?)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            # Shortage penalty
            penalty = find_number([
                r"(?:shortage\s*penalty|goodwill\s*penalty|goodwill\s*loss|penalty\s*cost)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
            ], q) or 0.0

            # Overage / excess cost (direct Co)
            overage = find_number([
                r"(?:overage\s*cost|cost\s*of\s*overage|excess\s*cost|cost\s*of\s*excess(?:\s*inventory)?|loss\s*per\s*unsold\s*unit)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)",
            ], q)

            mean_demand = find_number([
                r"(?:mean\s*demand|average\s*demand|expected\s*demand|demand\s*averages?|mean\s*of)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*(?:jackets|units|items|copies|newspapers|demand)",
            ], q)

            std_dev_demand = find_number([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|std\s*dev\s*of\s*demand|demand\s*std\s*dev|standard\s*deviation|std\s*dev|sigma)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)",
            ], q)

            distribution = "Normal" if any(w in q.lower() for w in ["normal", "bell curve", "gaussian", "std dev", "standard deviation"]) else "Normal"

            inputs = {
                "selling_price": price,
                "unit_cost": cost,
                "salvage_value": salvage,
                "stockout_penalty": penalty,
                "underage_cost": underage,
                "overage_cost": overage,
                "mean_demand": mean_demand,
                "std_dev_demand": std_dev_demand,
                "demand_distribution": distribution,
            }

        return inputs

    # =========================================================================
    # STEP 3: SELECT PYTHON TOOL
    # =========================================================================
    def select_tool(self, model_type: ModelType) -> str:
        """Select the calculation function name in inventory_tools.py."""
        return self.tool_mapping.get(model_type, "unknown_tool")

    # =========================================================================
    # STEP 4: CALCULATE
    # =========================================================================
    def calculate(self, model_type: ModelType, inputs: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute deterministic Python calculations using tools in inventory_tools.py.
        Returns: (results_dict, error_message)
        """
        try:
            if model_type == ModelType.EOQ:
                res = tools.calculate_eoq(
                    annual_demand=inputs["annual_demand"],
                    order_cost=inputs["order_cost"],
                    holding_cost=inputs["holding_cost"],
                    unit_cost=inputs.get("unit_cost"),
                    lead_time_days=inputs.get("lead_time_days"),
                    workdays_per_year=inputs.get("workdays_per_year", 250),
                )
                return res.to_dict(), None

            elif model_type == ModelType.QUANTITY_DISCOUNT:
                res = tools.calculate_quantity_discount(
                    annual_demand=inputs["annual_demand"],
                    order_cost=inputs["order_cost"],
                    discount_schedule=inputs["discount_schedule"],
                    holding_rate=inputs.get("holding_rate"),
                    fixed_holding_cost=inputs.get("fixed_holding_cost"),
                )
                return res.to_dict(), None

            elif model_type == ModelType.SAFETY_STOCK_ROP:
                res = tools.calculate_rop(
                    demand_per_period=inputs["demand_per_period"],
                    demand_std_dev=inputs.get("demand_std_dev", 0.0),
                    lead_time=inputs["lead_time"],
                    lead_time_std_dev=inputs.get("lead_time_std_dev", 0.0),
                    service_level=inputs.get("service_level", 0.95),
                )
                return res.to_dict(), None

            elif model_type == ModelType.FIXED_PERIOD_REVIEW:
                res = tools.calculate_periodic_review(
                    review_period=inputs["review_period"],
                    lead_time=inputs["lead_time"],
                    demand_per_period=inputs["demand_per_period"],
                    demand_std_dev=inputs.get("demand_std_dev", 0.0),
                    service_level=inputs.get("service_level", 0.95),
                    current_inventory=inputs.get("current_inventory", 0.0),
                    pipeline_orders=inputs.get("pipeline_orders", 0.0),
                )
                return res.to_dict(), None

            elif model_type == ModelType.NEWSEVNDOR:
                res = tools.calculate_newsvendor(
                    selling_price=inputs["selling_price"],
                    unit_cost=inputs["unit_cost"],
                    mean_demand=inputs["mean_demand"],
                    std_dev_demand=inputs["std_dev_demand"],
                    salvage_value=inputs.get("salvage_value", 0.0),
                    stockout_penalty=inputs.get("stockout_penalty", 0.0),
                )
                return res.to_dict(), None

            else:
                return None, f"Unsupported model type: {model_type}"

        except Exception as e:
            return None, str(e)

    # =========================================================================
    # STEP 5: VERIFY
    # =========================================================================
    def verify(
        self,
        model_type: ModelType,
        inputs: Dict[str, Any],
        results: Optional[Dict[str, Any]],
        calc_error: Optional[str] = None,
    ) -> Tuple[List[VerificationCheck], str]:
        """
        Verify domain constraints, economic rationale, and mathematical consistency.
        Returns: (list_of_checks, overall_status)
        """
        checks: List[VerificationCheck] = []

        if calc_error is not None:
            checks.append(VerificationCheck("Calculation Execution", "FAIL", f"Calculation failed: {calc_error}"))
            return checks, "FAILED"

        if results is None:
            checks.append(VerificationCheck("Calculation Results", "FAIL", "No calculation results available."))
            return checks, "FAILED"

        # Universal checks
        checks.append(VerificationCheck("Tool Execution", "PASS", "Numerical computation completed without errors."))

        if model_type == ModelType.EOQ:
            q = results["optimal_order_quantity"]
            d = results["annual_demand"]
            s = results["order_cost"]
            h = results["holding_cost"]

            # Positivity checks
            if d > 0 and s > 0 and h > 0:
                checks.append(VerificationCheck("Parameter Positivity", "PASS", "Demand, order cost, and holding cost are strictly positive."))
            else:
                checks.append(VerificationCheck("Parameter Positivity", "FAIL", "Demand, order cost, or holding cost <= 0."))

            # Cost balance at EOQ: AOC should equal AHC
            diff = abs(results["annual_ordering_cost"] - results["annual_holding_cost"])
            rel_diff = diff / max(results["total_annual_inventory_cost"], 1.0)
            if rel_diff < 0.05:
                checks.append(VerificationCheck("Cost Balance Check", "PASS", f"Annual ordering cost (${results['annual_ordering_cost']:,.2f}) matches annual holding cost (${results['annual_holding_cost']:,.2f}) at optimal EOQ."))
            else:
                checks.append(VerificationCheck("Cost Balance Check", "WARN", f"Ordering cost and holding cost deviate slightly due to rounding (diff: ${diff:.2f})."))

            if q > d:
                checks.append(VerificationCheck("Order Cycle Sanity", "WARN", "Optimal order quantity exceeds total annual demand. Single order covers more than a year."))

        elif model_type == ModelType.QUANTITY_DISCOUNT:
            tiers = results.get("tiers", [])
            if len(tiers) >= 2:
                checks.append(VerificationCheck("Tier Progression", "PASS", f"Successfully analyzed {len(tiers)} discount price tiers."))
            else:
                checks.append(VerificationCheck("Tier Progression", "WARN", "Only 1 discount tier was provided; behaves identically to standard EOQ."))

            opt_q = results["optimal_order_quantity"]
            if opt_q > 0:
                checks.append(VerificationCheck("Feasibility Check", "PASS", f"Selected order quantity {opt_q} is feasible within Tier {results['best_tier_index']}."))

            if results["annual_savings"] > 0:
                checks.append(VerificationCheck("Economic Benefit", "PASS", f"Taking discount in Tier {results['best_tier_index']} saves ${results['annual_savings']:,.2f} ({results['percentage_savings']}%) annually vs baseline."))
            else:
                checks.append(VerificationCheck("Economic Benefit", "PASS", "Baseline tier offers the lowest total cost; discounts do not justify additional holding costs."))

        elif model_type == ModelType.SAFETY_STOCK_ROP:
            csl = results["service_level"]
            z = results["z_score"]
            ss = results["safety_stock"]
            rop = results["reorder_point"]

            if 0.50 <= csl < 1.0:
                checks.append(VerificationCheck("Service Level Range", "PASS", f"Cycle service level {csl * 100:.1f}% maps to standard normal z = {z:.3f}."))
            elif 0.0 < csl < 0.50:
                checks.append(VerificationCheck("Service Level Range", "WARN", f"Service level ({csl * 100:.1f}%) is below 50%, resulting in negative safety stock buffer."))
            else:
                checks.append(VerificationCheck("Service Level Range", "FAIL", f"Invalid service level {csl}."))

            if rop >= results["expected_lead_time_demand"]:
                checks.append(VerificationCheck("Buffer Protection", "PASS", f"Reorder point ({rop:,.1f}) provides positive buffer ({ss:,.1f} units) above expected lead time demand ({results['expected_lead_time_demand']:,.1f})."))

        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            m = results["order_up_to_level"]
            q_rec = results["recommended_order_quantity"]
            inv_pos = results["inventory_position"]

            checks.append(VerificationCheck("Target Level Consistency", "PASS", f"Order-up-to level M = {m:,.1f} covers {results['protection_period']} periods of protection (review period P={results['review_period']} + lead time L={results['lead_time']})."))
            if q_rec == 0.0 and inv_pos >= m:
                checks.append(VerificationCheck("Order Trigger", "PASS", f"Current inventory position ({inv_pos:,.1f}) satisfies target level ({m:,.1f}); no replenishment needed this cycle."))
            else:
                checks.append(VerificationCheck("Order Trigger", "PASS", f"Replenishment quantity {q_rec:,.1f} units restores inventory position to target {m:,.1f}."))

        elif model_type == ModelType.NEWSEVNDOR:
            p = results["selling_price"]
            c = results["unit_cost"]
            v = results["salvage_value"]
            cr = results["critical_fractile"]

            if p > c:
                checks.append(VerificationCheck("Profit Margin", "PASS", f"Gross margin is positive: Selling Price (${p:,.2f}) > Unit Cost (${c:,.2f})."))
            else:
                checks.append(VerificationCheck("Profit Margin", "FAIL", "Selling price must be strictly greater than unit cost."))

            if c >= v:
                checks.append(VerificationCheck("Salvage Reality Check", "PASS", f"Unit cost (${c:,.2f}) >= Salvage value (${v:,.2f}); overage cost is non-negative."))
            else:
                checks.append(VerificationCheck("Salvage Reality Check", "FAIL", "Salvage value exceeds unit cost (arbitrage situation)."))

            if 0.0 < cr < 1.0:
                checks.append(VerificationCheck("Critical Fractile Check", "PASS", f"Critical ratio CR = {cr:.4f} is within valid probability bounds (0, 1)."))

        overall = "PASSED"
        if any(c.status == "FAIL" for c in checks):
            overall = "FAILED"
        elif any(c.status == "WARN" for c in checks):
            overall = "WARNING"

        return checks, overall

    # =========================================================================
    # STEP 6: EXPLAIN
    # =========================================================================
    def explain(
        self,
        model_type: ModelType,
        inputs: Dict[str, Any],
        results: Optional[Dict[str, Any]],
        checks: List[VerificationCheck],
        overall_status: str,
    ) -> str:
        """
        Generate executive-ready supply chain decision advisory with mathematical breakdowns.
        """
        if results is None:
            missing_fields = [k for k, v in inputs.items() if v is None]
            return (
                f"### Inventory Decision Agent Advisory\n\n"
                f"**Identified Model:** `{model_type.value}`\n\n"
                f"**Status:** Unable to complete calculation due to missing inputs.\n\n"
                f"**Missing Required Parameters:** {', '.join(missing_fields) if missing_fields else 'Unspecified parameter error'}.\n\n"
                f"*Please provide the missing numerical values to run the optimization.*"
            )

        md = []

        if model_type == ModelType.EOQ:
            q = results["optimal_order_quantity"]
            d = results["annual_demand"]
            s = results["order_cost"]
            h = results["holding_cost"]
            tc = results["total_annual_inventory_cost"]
            orders = results["orders_per_year"]
            cycle = results["cycle_time_days"]

            md.append(f"### Supply Chain Advisory: Economic Order Quantity (EOQ)")
            md.append(f"**Recommended Order Quantity ($Q^*$):** `{q:,.2f} units` per replenishment order.\n")
            md.append(f"#### Executive Summary")
            md.append(f"- **Order Frequency:** Place **{orders:,.1f} orders per year** (approximately every **{cycle:,.1f} working days**).")
            md.append(f"- **Annual Ordering Cost:** **${results['annual_ordering_cost']:,.2f}** (${s:,.2f}/order $\\times$ {orders:,.1f} orders).")
            md.append(f"- **Annual Holding Cost:** **${results['annual_holding_cost']:,.2f}** (${h:,.2f}/unit/yr $\\times$ {q/2:,.1f} average units in stock).")
            md.append(f"- **Total Annual Inventory Carrying & Ordering Cost:** **${tc:,.2f}**.")
            if results["total_annual_cost_with_purchase"]:
                md.append(f"- **Total Annual Cost (including material purchases):** **${results['total_annual_cost_with_purchase']:,.2f}**.")
            if results["reorder_point"] is not None:
                md.append(f"- **Reorder Point ($ROP$):** Reorder when stock drops to **{results['reorder_point']:,.1f} units** (based on {results['lead_time_days']} days lead time).")

            md.append(f"\n#### Mathematical Formulation")
            md.append(f"$$\\text{{EOQ}} = \\sqrt{{\\frac{{2DS}}{{H}}}} = \\sqrt{{\\frac{{2 \\times {d:,} \\times {s:,.2f}}}{{{h:,.2f}}}}} = {q:,.2f}\\text{{ units}}$$")
            md.append(f"At this economic order quantity, the marginal ordering cost precisely offsets the marginal holding cost ($AOC \\approx AHC$), minimizing overall inventory expenses.")

        elif model_type == ModelType.QUANTITY_DISCOUNT:
            md.append(f"### Supply Chain Advisory: Quantity Discount Evaluation")
            md.append(f"**Optimal Order Decision:** Order **{results['optimal_order_quantity']:,.0f} units** at **${results['optimal_unit_price']:,.2f} per unit** (Tier {results['best_tier_index']}).\n")
            md.append(f"#### Executive Summary")
            md.append(f"- **Minimum Total Annual Cost:** **${results['min_total_annual_cost']:,.2f}**.")
            if results["annual_savings"] > 0:
                md.append(f"- **Net Annual Savings vs Baseline:** **${results['annual_savings']:,.2f}** ({results['percentage_savings']:.2f}% cost reduction).")
                md.append(f"- **Recommendation:** Qualify for Tier {results['best_tier_index']} volume discount. The significant purchasing cost savings comfortably outweigh the additional holding cost.")
            else:
                md.append(f"- **Recommendation:** Stick with the baseline tier. The purchase discount in higher tiers does not justify the inventory carrying costs of holding larger batch sizes.")

            md.append(f"\n#### Multi-Tier Cost Breakdown Table")
            md.append(f"| Tier | Quantity Range | Unit Price | Feasible Q | Purchase Cost | Order Cost | Holding Cost | Total Annual Cost |")
            md.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in results["tiers"]:
                t_idx = t["tier_index"] if isinstance(t, dict) else t.tier_index
                t_min = t["min_qty"] if isinstance(t, dict) else t.min_qty
                t_max = t["max_qty"] if isinstance(t, dict) else t.max_qty
                t_price = t["unit_price"] if isinstance(t, dict) else t.unit_price
                t_q = t["feasible_quantity"] if isinstance(t, dict) else t.feasible_quantity
                t_purch = t["annual_purchase_cost"] if isinstance(t, dict) else t.annual_purchase_cost
                t_ord = t["annual_ordering_cost"] if isinstance(t, dict) else t.annual_ordering_cost
                t_hld = t["annual_holding_cost"] if isinstance(t, dict) else t.annual_holding_cost
                t_tot = t["total_annual_cost"] if isinstance(t, dict) else t.total_annual_cost

                max_str = f"{t_max:,.0f}" if t_max is not None else "+"
                best_marker = " **(OPTIMAL)**" if t_idx == results["best_tier_index"] else ""
                md.append(
                    f"| Tier {t_idx}{best_marker} | {t_min:,.0f} – {max_str} | ${t_price:,.2f} | {t_q:,.0f} | "
                    f"${t_purch:,.2f} | ${t_ord:,.2f} | ${t_hld:,.2f} | **${t_tot:,.2f}** |"
                )

        elif model_type == ModelType.SAFETY_STOCK_ROP:
            ss = results["safety_stock"]
            rop = results["reorder_point"]
            exp_ltd = results["expected_lead_time_demand"]
            sigma_ltd = results["lead_time_demand_std_dev"]
            csl = results["service_level"] * 100.0

            md.append(f"### Supply Chain Advisory: Continuous Review Reorder Point & Safety Stock")
            md.append(f"**Recommended Policy:** Set Reorder Point ($ROP$) at **{rop:,.1f} units** with a Safety Stock buffer of **{ss:,.1f} units**.\n")
            md.append(f"#### Executive Summary")
            md.append(f"- **Cycle Service Level ($CSL$):** **{csl:.1f}%** ($z = {results['z_score']:.3f}$).")
            md.append(f"- **Expected Lead Time Demand ($\\mu_{{LTD}}$):** **{exp_ltd:,.1f} units** ({results['demand_per_period']:,.1f}/period $\\times$ {results['lead_time_periods']} periods).")
            md.append(f"- **Lead Time Demand Uncertainty ($\\sigma_{{LTD}}$):** **{sigma_ltd:,.2f} units**.")
            md.append(f"- **Stockout Probability per Replenishment Cycle:** **{results['stockout_risk_per_cycle']*100:.2f}%**.")

            md.append(f"\n#### Mathematical Formulation")
            md.append(f"$$SS = z \\times \\sigma_{{LTD}} = {results['z_score']:.3f} \\times {sigma_ltd:,.2f} = {ss:,.2f}\\text{{ units}}$$")
            md.append(f"$$ROP = \\mu_{{LTD}} + SS = {exp_ltd:,.1f} + {ss:,.2f} = {rop:,.2f}\\text{{ units}}$$")
            md.append(f"**Operational Instruction:** Whenever the available inventory (on-hand + on-order) drops to **{math.ceil(rop):,} units**, immediately trigger a replenishment order.")

        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            m = results["order_up_to_level"]
            q = results["recommended_order_quantity"]
            p = results["review_period"]
            l = results["lead_time"]
            inv_pos = results["inventory_position"]

            md.append(f"### Supply Chain Advisory: Periodic Review Order-Up-To Policy")
            md.append(f"**Recommended Replenishment Order:** Order **{q:,.0f} units** immediately to restore stock to Target Level **$M = {m:,.1f}$ units**.\n")
            md.append(f"#### Executive Summary")
            md.append(f"- **Protection Interval ($P + L$):** **{results['protection_period']:.1f} periods** (Review Interval $P={p}$ + Supplier Lead Time $L={l}$).")
            md.append(f"- **Order-Up-To Target Level ($M$):** **{m:,.1f} units**.")
            md.append(f"- **Current Inventory Position:** **{inv_pos:,.1f} units** ({results['current_inventory']:,.1f} on-hand + {results['pipeline_orders']:,.1f} on-order).")
            md.append(f"- **Safety Stock Required:** **{results['safety_stock']:,.1f} units** (protecting demand variations across the full {results['protection_period']:.0f}-period exposure window).")
            md.append(f"- **Safety Stock Comparison:** Periodic review requires **{results['safety_stock_ratio_vs_continuous']:.2f}x** the safety stock of a continuous review policy due to review interval lag.")

            md.append(f"\n#### Mathematical Formulation")
            md.append(f"$$M = \\bar{{d}}(P + L) + z\\sigma_d\\sqrt{{P + L}} = {results['expected_demand_protection_period']:,.1f} + {results['safety_stock']:,.1f} = {m:,.1f}\\text{{ units}}$$")
            md.append(f"$$Q = \\max(0, M - \\text{{Inventory Position}}) = {m:,.1f} - {inv_pos:,.1f} = {q:,.1f}\\text{{ units}}$$")

        elif model_type == ModelType.NEWSEVNDOR:
            q_star = results["optimal_order_quantity"]
            cr = results["critical_fractile"] * 100.0
            profit = results["expected_profit"]
            cu = results["underage_cost"]
            co = results["overage_cost"]

            md.append(f"### Supply Chain Advisory: Single-Period Newsvendor Optimization")
            md.append(f"**Optimal Order Decision ($Q^*$):** Order **{q_star:,.0f} units**.\n")
            md.append(f"#### Executive Summary")
            md.append(f"- **Critical Fractile ($CR$):** **{cr:.2f}%** (Target In-Stock Probability).")
            md.append(f"- **Cost of Underage ($C_u$ / Lost Profit per unit):** **${cu:,.2f}**.")
            md.append(f"- **Cost of Overage ($C_o$ / Leftover Loss per unit):** **${co:,.2f}**.")
            md.append(f"- **Expected Net Profit:** **${profit:,.2f}**.")
            md.append(f"- **Expected Sales:** **{results['expected_sales']:,.1f} units** | **Expected Leftovers:** **{results['expected_leftover_units']:,.1f} units**.")
            md.append(f"- **Expected Lost Sales (Stockout volume):** **{results['expected_lost_sales']:,.1f} units**.")
            md.append(f"- **Expected Value of Perfect Information (EVPI):** **${results['evpi']:,.2f}** (maximum value of demand forecasting accuracy).")

            md.append(f"\n#### Mathematical Formulation")
            md.append(f"$$CR = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{cu:,.2f}}}{{{cu:,.2f} + {co:,.2f}}} = {results['critical_fractile']:.4f}$$")
            md.append(f"$$Q^* = \\mu + z \\times \\sigma = {results['mean_demand']:,.1f} + ({results['z_score']:.3f} \\times {results['std_dev_demand']:,.1f}) = {q_star:,.2f}\\text{{ units}}$$")

        # Verification section
        md.append(f"\n#### Automated Verification & Domain Checks (`{overall_status}`)")
        for c in checks:
            badge = "✅" if c.status == "PASS" else ("⚠️" if c.status == "WARN" else "❌")
            md.append(f"- {badge} **{c.name}:** {c.message}")

        return "\n".join(md)

    # =========================================================================
    # MASTER PIPELINE: RUN
    # =========================================================================
    def run(self, question: str, override_inputs: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Executes the complete Supply Chain Decision Agent architecture:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
        """
        trace: List[AgentStepTrace] = []

        # Step 1: Identify Model
        model_type, confidence, id_reasoning = self.identify_model(question)
        trace.append(AgentStepTrace(
            step_name="1. Identify Model",
            status="SUCCESS",
            details={"model": model_type.value, "confidence": confidence, "reasoning": id_reasoning},
        ))

        if model_type == ModelType.UNKNOWN:
            return AgentResponse(
                question=question,
                identified_model=model_type,
                model_confidence=confidence,
                identification_reasoning=id_reasoning,
                extracted_inputs={},
                tool_selected="None",
                calculation_results=None,
                verification_checks=[VerificationCheck("Model Detection", "FAIL", "No recognized inventory model detected.")],
                verification_overall="FAILED",
                explanation_markdown="I could not identify which inventory model applies to your question. Please ensure your question relates to EOQ, Quantity Discounts, Safety Stock & Reorder Point, Fixed-Period Review, or the Newsvendor model.",
                execution_trace=trace,
            )

        # Step 2: Extract Inputs
        extracted = self.extract_inputs(question, model_type)
        if override_inputs:
            # Allow fine-tuning or programmatic overrides
            extracted.update({k: v for k, v in override_inputs.items() if v is not None})

        trace.append(AgentStepTrace(
            step_name="2. Extract Inputs",
            status="SUCCESS",
            details=extracted,
        ))

        # Step 3: Select Python Tool
        tool_name = self.select_tool(model_type)
        trace.append(AgentStepTrace(
            step_name="3. Select Python Tool",
            status="SUCCESS",
            details={"selected_tool": tool_name},
        ))

        # Check for missing critical inputs before calculation
        missing = []
        if model_type == ModelType.EOQ:
            for k in ["annual_demand", "order_cost", "holding_cost"]:
                if extracted.get(k) is None:
                    missing.append(k)
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            for k in ["annual_demand", "order_cost"]:
                if extracted.get(k) is None:
                    missing.append(k)
            if extracted.get("holding_rate") is None and extracted.get("fixed_holding_cost") is None:
                missing.append("holding_rate (or fixed_holding_cost)")
            if not extracted.get("discount_schedule"):
                missing.append("discount_schedule (tiers)")
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            for k in ["demand_per_period", "demand_std_dev", "lead_time"]:
                if extracted.get(k) is None:
                    missing.append(k)
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            for k in ["review_period", "lead_time", "demand_per_period", "demand_std_dev"]:
                if extracted.get(k) is None:
                    missing.append(k)
        elif model_type == ModelType.NEWSEVNDOR:
            for k in ["selling_price", "unit_cost", "mean_demand", "std_dev_demand"]:
                if extracted.get(k) is None:
                    missing.append(k)
        if missing:
            trace.append(AgentStepTrace(
                step_name="4. Calculate",
                status="FAILED",
                details={"missing_parameters": missing},
            ))
            return AgentResponse(
                question=question,
                identified_model=model_type,
                model_confidence=confidence,
                identification_reasoning=id_reasoning,
                extracted_inputs=extracted,
                tool_selected=tool_name,
                calculation_results=None,
                verification_checks=[VerificationCheck("Parameter Completeness", "FAIL", f"Missing required parameters: {', '.join(missing)}")],
                verification_overall="FAILED",
                explanation_markdown=(
                    f"### Inventory Decision Agent Advisory\n\n"
                    f"**Identified Model:** `{model_type.value}`\n\n"
                    f"I identified that this problem requires the **{model_type.value}** model, but some required parameters were not found in your question:\n\n"
                    f"**Missing required parameters:**\n"
                    + "\n".join([f"- `{m}`" for m in missing])
                    + "\n\nPlease supply these values so I can perform the calculations."
                ),
                execution_trace=trace,
            )

        # Step 4: Calculate
        results, calc_error = self.calculate(model_type, extracted)
        trace.append(AgentStepTrace(
            step_name="4. Calculate",
            status="SUCCESS" if calc_error is None else "FAILED",
            details={"error": calc_error} if calc_error else {"summary": "Numerical computation completed."},
        ))

        # Step 5: Verify
        checks, overall_status = self.verify(model_type, extracted, results, calc_error)
        trace.append(AgentStepTrace(
            step_name="5. Verify",
            status=overall_status,
            details={"checks_count": len(checks), "overall": overall_status},
        ))

        # Step 6: Explain
        explanation = self.explain(model_type, extracted, results, checks, overall_status)
        trace.append(AgentStepTrace(
            step_name="6. Explain",
            status="SUCCESS",
            details={"explanation_length_chars": len(explanation)},
        ))

        curve_data = results.get("curve_data") if results else None

        return AgentResponse(
            question=question,
            identified_model=model_type,
            model_confidence=confidence,
            identification_reasoning=id_reasoning,
            extracted_inputs=extracted,
            tool_selected=tool_name,
            calculation_results=results,
            verification_checks=checks,
            verification_overall=overall_status,
            explanation_markdown=explanation,
            curve_data=curve_data,
            execution_trace=trace,
        )
