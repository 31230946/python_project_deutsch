# core is a regular Python package (has __init__.py) that holds every
# building block of the "Should I Buy This?" agent:
#
#   models.py       -> Product / DecisionRecord classes (OOP)
#   history.py      -> file handling + custom iterator over past decisions
#   rules_engine.py -> deterministic rule checks + shallow vs deep copy demo
#   llm_fallback.py -> generator that yields Gemini key(s) then Ollama
#   guardrails.py   -> jailbreak / prompt-injection resistance
#   tools.py        -> the four @tool functions the agent can call
#   agent.py        -> builds the LangChain agent + structured-output verdict
#
# Keeping each concern in its own module is exactly the "custom modules /
# package structure" part of the syllabus, and it is not just for show --
# app.py genuinely imports from each of these.
