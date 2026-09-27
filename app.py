"""
Streamlit UI for the "Should I Buy This?" Decision Agent.

Run with:
    streamlit run app.py

Layout:
- Sidebar: current rules profile (read from data/rules.json) + a debug
  expander (dir(), shallow-vs-deep copy demo, custom for-loop / range demo)
- Main: paste a product description -> agent evaluates it live, with tool
  calls printed as they happen -> structured verdict -> decision history
"""

import json
import os

import streamlit as st
from langchain.messages import AIMessage, AIMessageChunk

from core.agent import build_agent, build_user_turn_message, extract_product, get_verdict
from core.fundamentals_demo import custom_for_loop, introspect, my_range
from core.guardrails import screen_input
from core.history import DecisionHistory, buy_ratio_last_n
from core.llm_fallback import get_working_model
from core.models import DecisionRecord
from core.rules_engine import (
    RULES_PATH,
    check_budget_rule,
    check_features_rule,
    demonstrate_copy_bug,
    load_rules,
)

st.set_page_config(page_title="Should I Buy This?", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------------
# Token / cost estimate helper (item 8 from the spec: small, real,
# ties to the token-cost lesson -- not a precise billing tool, just an
# honest approximation shown next to each LLM call).
# ---------------------------------------------------------------------
APPROX_CHARS_PER_TOKEN = 4
APPROX_COST_PER_1K_TOKENS_INR = 0.15  # rough Gemini Flash-tier estimate


def estimate_tokens_and_cost(text: str):
    tokens = max(1, len(text) // APPROX_CHARS_PER_TOKEN)
    cost = (tokens / 1000) * APPROX_COST_PER_1K_TOKENS_INR
    return tokens, cost


# ---------------------------------------------------------------------
# Sidebar: rules profile + debug expander
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Your Rules Profile")
    st.caption(f"Loaded from `{os.path.relpath(RULES_PATH)}` -- edit that file to change your profile.")
    try:
        rules = load_rules()
        st.json(rules)
    except RuntimeError as e:
        st.error(str(e))
        rules = {}

    st.divider()
    st.subheader("Decision history")
    st.caption(buy_ratio_last_n(5))
    history_records = list(DecisionHistory())
    if history_records:
        for record in history_records[:5]:
            st.text(record.summary())
    else:
        st.text("No decisions logged yet.")

    st.divider()
    with st.expander("Debug / introspection"):
        st.markdown("**Shallow vs deep copy demo** (`core/rules_engine.py`)")
        if st.button("Run copy-semantics demo"):
            st.code(demonstrate_copy_bug())

        st.markdown("**`dir()` on a config object**")
        st.code("\n".join(introspect(rules)[:15]))

        st.markdown("**Custom for-loop (built from `while` + `next()`)**")
        st.code(f"custom_for_loop('buy') -> {custom_for_loop('buy')}")

        st.markdown("**`range()` reimplemented as a generator**")
        st.code(f"list(my_range(0, 10, 2)) -> {list(my_range(0, 10, 2))}")

        st.markdown(
            "This block itself only runs inside Streamlit's script re-run, "
            "but `core/fundamentals_demo.py` also has an "
            "`if __name__ == \"__main__\":` guard at the bottom, so running "
            "`python core/fundamentals_demo.py` directly prints these same "
            "demos on the command line without importing Streamlit at all."
        )

# ---------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------
st.title("🛒 Should I Buy This?")
st.caption("Describe a product + price in plain English. A real LangChain agent will check it against your rules and give a verdict.")

product_text = st.text_area(
    "Product description",
    placeholder="e.g. 'Thinking of buying a Samsung soundbar for around 12000, has a 1 year warranty and bluetooth'",
    height=120,
)

col1, col2 = st.columns([1, 1])
evaluate_clicked = col1.button("Evaluate", type="primary")
clear_clicked = col2.button("Clear")

if clear_clicked:
    st.rerun()

if evaluate_clicked:
    flagged, reason = screen_input(product_text)
    if flagged:
        st.warning(
            f"Input was not sent to the agent: {reason}\n\n"
            "This is the first guardrail layer -- obviously instruction-like "
            "or off-topic input gets stopped before it ever reaches the LLM."
        )
        st.stop()

    # ---- connect to a model, with visible fallback -------------------
    with st.spinner("Connecting to a model (Gemini, then Ollama fallback if needed)..."):
        try:
            label, model = get_working_model()
        except RuntimeError as e:
            st.error(f"Could not reach any model: {e}")
            st.stop()
    st.success(f"Using model: {label}")

    tokens_in, cost_in = estimate_tokens_and_cost(product_text)
    st.caption(f"Approx input tokens: {tokens_in} | Estimated cost: Rs.{cost_in:.4f} (rough estimate, not exact billing)")

    # ---- LLM use #1: structured extraction ---------------------------
    try:
        with st.spinner("Extracting structured product data..."):
            product = extract_product(model, product_text)
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        st.stop()

    st.subheader("Extracted product")
    st.write(f"**{product.name}** — Rs.{product.price:.0f} — category: `{product.category}`")
    st.write(f"Features detected ({len(product)}): {', '.join(product.features) if product.features else 'none'}")

    # ---- deterministic rule checks (these are what the agent's tools
    #      also compute internally -- shown here directly too, for
    #      transparency, since these are the real decision parameters) --
    budget_result = check_budget_rule(product.price, product.category)
    feature_result = check_features_rule(product.features)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.metric("Budget check", "Within budget" if budget_result["fits_budget"] else "Over budget")
        st.caption(f"Effective cap: Rs.{budget_result['effective_cap']:.0f}")
    with rc2:
        st.metric("Feature check", "OK" if feature_result["must_haves_satisfied"] and not feature_result["has_dealbreaker"] else "Issue found")
        if feature_result["missing_must_haves"]:
            st.caption(f"Missing must-haves: {', '.join(feature_result['missing_must_haves'])}")
        if feature_result["has_dealbreaker"]:
            st.caption(f"Dealbreakers found: {', '.join(feature_result['found_dealbreakers'])}")

    # ---- LLM use #2: the real agent, tool calls streamed live --------
    st.subheader("Agent reasoning (live tool calls)")
    agent = build_agent(model)
    user_message = build_user_turn_message(product_text, product)

    tool_call_box = st.empty()
    answer_box = st.empty()
    tool_call_log = []
    ai_message_text = ""

    try:
        for chunk, metadata in agent.stream(
            {"messages": [user_message]},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk):
                ai_message_text += chunk.content or ""
                answer_box.markdown(ai_message_text)

            for tool_call in getattr(chunk, "tool_call_chunks", []) or []:
                tool_name = tool_call.get("name")
                if tool_name:
                    tool_call_log.append(f"🔧 Calling `{tool_name}`...")
                    tool_call_box.markdown("\n\n".join(tool_call_log))
    except Exception as e:
        st.error(f"Agent run failed: {e}")
        st.stop()

    # ---- LLM use #3: structured verdict, gated by the rule results ---
    try:
        verdict = get_verdict(model, product, budget_result, feature_result)
    except Exception as e:
        st.error(f"Could not produce a structured verdict: {e}")
        st.stop()

    st.subheader("Verdict")
    decision = verdict["decision"]
    color = {"Buy": "green", "Don't Buy": "red", "Consider": "orange"}.get(decision, "blue")
    st.markdown(f"### :{color}[{decision}]  \n**Confidence:** {verdict['confidence']}")
    st.write(verdict["reason"])

    # log the decision to history (also demonstrates the log_decision tool
    # path even if the agent didn't happen to call it itself this turn)
    from core.history import append_decision
    append_decision(DecisionRecord(
        product_name=product.name,
        price=product.price,
        decision=decision,
        reason=verdict["reason"],
        confidence=verdict["confidence"],
    ))

    tokens_out, cost_out = estimate_tokens_and_cost(ai_message_text + verdict["reason"])
    st.caption(f"Approx output tokens: {tokens_out} | Estimated cost: Rs.{cost_out:.4f}")

    st.info("Decision logged. Check the sidebar's Decision history section.")
