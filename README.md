# Should I Buy This? — Decision Agent

A LangChain **agent** (not a scripted chatbot) that decides whether a
described purchase fits your saved rules, using real tool-calling.

## Project structure

```
buy_decision_agent/
├── app.py                    Streamlit UI, orchestrates everything
├── requirements.txt
├── .env.example               copy to .env and fill in your key(s)
├── data/
│   ├── rules.json              your editable "profile" (budget, must-haves, dealbreakers)
│   └── decision_log.txt        created automatically, append-only history
└── core/                       package: __init__.py + these modules
    ├── models.py                Product / DecisionRecord (OOP, ABC, operator overloading)
    ├── rules_engine.py          deterministic budget/feature checks + shallow vs deep copy demo
    ├── history.py                file handling (x/a/r modes) + custom iterator class
    ├── llm_fallback.py           generator-based Gemini -> Gemini(backup) -> Ollama fallback
    ├── guardrails.py             jailbreak / prompt-injection input screening
    ├── tools.py                   the 4 @tool functions the agent can call
    ├── agent.py                   create_agent() wiring + structured-output schemas
    └── fundamentals_demo.py       small extras: custom for-loop, range() as a generator, dir()
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GEMINI_API_KEY

# optional, for the offline fallback to actually work:
ollama pull qwen2.5:3b
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, type a product description
(e.g. *"Samsung soundbar, around 12000, has a 1 year warranty and
bluetooth"*), and click **Evaluate**.

## Editing your "profile"

Your profile is just `data/rules.json` — there is no hidden AI
personality model. Open it and change:

- `monthly_budget_ceiling` — your overall monthly spending cap
- `category_budget_caps` — per-category caps (electronics, clothing, ...)
- `must_have_keywords` — features that must be present
- `dealbreaker_keywords` — features that auto-flag a product

See `VIVA_NOTES.md` for the plain-language explanations to give in the
viva, and exactly where each syllabus concept lives in the code.
