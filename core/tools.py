"""
The agent's tools, using the exact `@tool` decorator pattern from class.

These are the ONLY things the agent can "do" in the outside world -- read
the rules file, check features, append to the decision log, and read the
decision log back. No shell access, no arbitrary file access, no network
calls. That containment is part of the jailbreak-resistance story: even
if someone talked the agent into misbehaving, there is nothing dangerous
within its reach.

IMPORTANT: the agent DECIDES on its own whether/when to call these, based
on the user's message -- that's what makes this an agent instead of a
fixed if/else pipeline (see core/agent.py's system prompt).
"""

from langchain.tools import tool

from core.history import append_decision, buy_ratio_last_n, read_all_decisions
from core.models import DecisionRecord
from core.rules_engine import check_budget_rule, check_features_rule


@tool
def check_budget(price: float, category: str = "general") -> str:
    """check_budget(price, category) takes the extracted price of a
    product (and optionally its category, e.g. 'electronics', 'clothing')
    and returns whether it fits inside the user's saved budget rules
    (rules.json). This is a deterministic check, not an AI guess."""
    try:
        result = check_budget_rule(price, category)
    except Exception as e:
        return f"Could not check budget: {e}"

    if result["fits_budget"]:
        return (
            f"WITHIN BUDGET: price Rs.{result['price']:.0f} is within the "
            f"effective cap of Rs.{result['effective_cap']:.0f} "
            f"(category cap: {result['category_cap']}, monthly ceiling: {result['monthly_budget_ceiling']})."
        )
    return (
        f"OVER BUDGET: price Rs.{result['price']:.0f} exceeds the effective "
        f"cap of Rs.{result['effective_cap']:.0f} "
        f"(category cap: {result['category_cap']}, monthly ceiling: {result['monthly_budget_ceiling']})."
    )


@tool
def check_features(features: list) -> str:
    """check_features(features) takes the list of extracted product
    features/keywords and checks them against the saved must-have and
    dealbreaker keyword lists in rules.json. Returns a plain-text summary
    of what's satisfied and what's missing/flagged."""
    try:
        result = check_features_rule(features)
    except Exception as e:
        return f"Could not check features: {e}"

    lines = []
    if result["must_haves_satisfied"]:
        lines.append("All must-have features are present.")
    else:
        lines.append(f"MISSING must-have keyword(s): {', '.join(result['missing_must_haves'])}.")

    if result["has_dealbreaker"]:
        lines.append(f"DEALBREAKER FOUND: {', '.join(result['found_dealbreakers'])}.")
    else:
        lines.append("No dealbreaker keywords found.")

    return " ".join(lines)


@tool
def log_decision(
    product: str,
    verdict: str,
    price: float = 0.0,
    confidence: str = "Medium",
    reason: str = "",
) -> str:
    """log_decision(product, verdict, price, confidence, reason) appends
    this decision to the local decision history file, so it can be looked
    back on later with get_history(). This is real file handling ('a'
    append mode under the hood), not an in-memory-only log."""
    try:
        record = DecisionRecord(
            product_name=product,
            price=price,
            decision=verdict,
            reason=reason,
            confidence=confidence,
        )
        append_decision(record)
    except Exception as e:
        return f"Could not log decision: {e}"
    finally:
        pass
    return f"Logged: {product} -> {verdict}."


@tool
def get_history() -> str:
    """get_history() reads back all previously logged decisions ('r' mode)
    and also reports how many of the last few decisions were 'Buy', using
    the custom DecisionHistoryIterator to walk the log."""
    try:
        records = read_all_decisions()
    except Exception as e:
        return f"Could not read history: {e}"

    if not records:
        return "No past decisions logged yet."

    recent = records[-5:]
    lines = [r.summary() for r in reversed(recent)]
    ratio_line = buy_ratio_last_n(5)
    return "Recent decisions:\n" + "\n".join(lines) + f"\n\n{ratio_line}"
