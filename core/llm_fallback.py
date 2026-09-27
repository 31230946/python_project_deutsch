"""
API-key / model fallback, built as a GENERATOR (yield), reusing the exact
"try the next available key/model" pattern from class.

candidate_models() doesn't build every model up front -- it LAZILY yields
one (label, factory) pair at a time. get_working_model() pulls from the
generator and only actually constructs + test-calls a model when it's
that candidate's turn. If Gemini key 1 fails (missing, invalid, quota
exceeded...), we move to Gemini key 2 if one is configured, and finally
fall back to the local Ollama model -- entirely offline, no API key
needed at all.
"""

import os
from typing import Callable, Generator, Tuple

from dotenv import load_dotenv

load_dotenv()  # never hardcode keys -- always read from .env / environment

ModelFactory = Callable[[], object]


def candidate_models() -> Generator[Tuple[str, ModelFactory], None, None]:
    """Yields (label, factory) pairs lazily, one at a time."""

    gemini_key_1 = os.getenv("GEMINI_API_KEY")
    gemini_key_2 = os.getenv("GEMINI_API_KEY_BACKUP")

    if gemini_key_1:
        def make_primary():
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=gemini_key_1,
                temperature=0.2,
            )
        yield ("Gemini (primary key)", make_primary)

    if gemini_key_2:
        def make_backup():
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_key_2,
                temperature=0.2,
            )
        yield ("Gemini (backup key)", make_backup)

    # Local, offline, no API key required -- always the last resort.
    def make_ollama():
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5:3b")
    yield ("Ollama qwen2.5:3b (local fallback)", make_ollama)


def get_working_model():
    """Walks the generator above and try/except's its way through each
    candidate until one actually responds. Returns (label, model_instance).
    Raises RuntimeError only if every single candidate -- including the
    local Ollama fallback -- fails."""

    errors = []
    for label, factory in candidate_models():
        try:
            model = factory()
            # cheap smoke test so a bad/expired key fails fast, right here,
            # instead of silently breaking mid-conversation later
            model.invoke("ping")
            return label, model
        except Exception as e:  # noqa: BLE001 -- intentionally broad: any
            # failure (missing key, network, quota, model not pulled
            # locally...) should just move on to the next candidate
            errors.append(f"{label} failed: {e}")
            continue
        finally:
            pass

    raise RuntimeError(
        "All model candidates failed:\n" + "\n".join(errors)
    )
