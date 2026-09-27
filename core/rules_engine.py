"""
This module is the "profile" from the instructor's Q1.

There is NO hidden LLM personality model here. The user's profile is just
this plain, editable rules.json file, and every decision parameter below
is computed with ordinary Python -- no AI guesswork involved. The LLM only
gets to explain the result afterwards (see core/agent.py).
"""

import copy
import json
import os
from typing import Dict, List

RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rules.json")
RULES_PATH = os.path.abspath(RULES_PATH)

# The "master" rules dict, loaded once when the app starts.
_master_rules: Dict = {}


def load_rules() -> Dict:
    """'r' mode read of the rules file. Cached in _master_rules so we don't
    re-read the disk on every single check."""
    global _master_rules
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            _master_rules = json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"rules.json not found at {RULES_PATH}. Create it first."
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"rules.json is not valid JSON: {e}") from e
    finally:
        pass
    return _master_rules


def get_rules_snapshot(deep: bool = True) -> Dict:
    """
    WHY THIS FUNCTION EXISTS (shallow vs deep copy, demonstrated for real):

    Every time we run a check, we want a per-run "working copy" of the
    rules -- for example the agent might want to temporarily relax a
    category cap for a single what-if question ("what if my electronics
    budget was 20000 instead of 15000?") without ever touching the actual
    saved rules.json / _master_rules.

    If we did a SHALLOW copy:

        working = copy.copy(_master_rules)   # or dict(_master_rules)
        working["category_budget_caps"]["electronics"] = 20000

    ...this LOOKS safe because `working` is a different dict object than
    `_master_rules`. But `category_budget_caps` is a nested dict, and a
    shallow copy only copies the OUTER dict -- the nested dict is still
    the *same object* in memory, shared by both. So the line above would
    silently corrupt the real rules too. Try it: with deep=False, the
    demo below will show master_rules changing when it shouldn't.

    A DEEP copy (copy.deepcopy) recursively copies nested dicts/lists too,
    so `working["category_budget_caps"]` is a brand new dict, and editing
    it never touches `_master_rules`. That's why we default to deep=True
    everywhere in the real rule-checking code.
    """
    if not _master_rules:
        load_rules()
    if deep:
        return copy.deepcopy(_master_rules)
    return copy.copy(_master_rules)


def demonstrate_copy_bug() -> str:
    """Small, self-contained proof used by the Streamlit debug expander --
    actually runs both copy strategies and reports what happened, instead
    of just claiming it in the viva."""
    if not _master_rules:
        load_rules()

    original_electronics_cap = _master_rules["category_budget_caps"]["electronics"]

    shallow = get_rules_snapshot(deep=False)
    shallow["category_budget_caps"]["electronics"] = 999999
    corrupted = _master_rules["category_budget_caps"]["electronics"] == 999999
    # undo the damage so the demo can be run repeatedly
    _master_rules["category_budget_caps"]["electronics"] = original_electronics_cap

    deep = get_rules_snapshot(deep=True)
    deep["category_budget_caps"]["electronics"] = 999999
    safe = _master_rules["category_budget_caps"]["electronics"] == original_electronics_cap

    return (
        f"Shallow copy test: master rules got corrupted = {corrupted} "
        f"(electronics cap leaked to 999999)\n"
        f"Deep copy test: master rules stayed safe = {safe} "
        f"(electronics cap is still {_master_rules['category_budget_caps']['electronics']})"
    )


def check_budget_rule(price: float, category: str) -> Dict:
    """Decision parameter 1: budget fit. Pure Python, hard rule."""
    rules = get_rules_snapshot(deep=True)
    ceiling = rules.get("monthly_budget_ceiling", float("inf"))
    category_cap = rules.get("category_budget_caps", {}).get(category.lower())

    effective_cap = min(ceiling, category_cap) if category_cap is not None else ceiling
    fits = price <= effective_cap

    return {
        "fits_budget": fits,
        "price": price,
        "effective_cap": effective_cap,
        "monthly_budget_ceiling": ceiling,
        "category_cap": category_cap,
    }


def check_features_rule(features: List[str]) -> Dict:
    """Decision parameters 2 & 3: must-have coverage + dealbreaker check.
    Both are plain keyword matching against the rules file -- no LLM
    involved in deciding pass/fail, only in explaining it afterwards."""
    rules = get_rules_snapshot(deep=True)
    must_haves = [m.lower() for m in rules.get("must_have_keywords", [])]
    dealbreakers = [d.lower() for d in rules.get("dealbreaker_keywords", [])]

    feature_text = " ".join(features).lower()

    missing_must_haves = [m for m in must_haves if m not in feature_text]
    found_dealbreakers = [d for d in dealbreakers if d in feature_text]

    return {
        "must_haves_satisfied": len(missing_must_haves) == 0,
        "missing_must_haves": missing_must_haves,
        "has_dealbreaker": len(found_dealbreakers) > 0,
        "found_dealbreakers": found_dealbreakers,
    }
