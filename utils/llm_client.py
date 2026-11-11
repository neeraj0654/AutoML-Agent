"""
LLM Client — wraps Google Gemini API with graceful fallback when no key is set.
"""
import os
import textwrap
from dotenv import load_dotenv

load_dotenv()

_client = None
_model = None
_available = False


def _init():
    global _client, _model, _available
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            _model = genai.GenerativeModel("gemini-1.5-flash")
            _available = True
        except Exception:
            _available = False
    else:
        _available = False


_init()


def is_available() -> bool:
    return _available


def ask(prompt: str, temperature: float = 0.4) -> str:
    """
    Send a prompt to Gemini and return the text response.
    Falls back to a placeholder message if LLM is unavailable.
    """
    if not _available:
        return _fallback(prompt)
    try:
        import google.generativeai as genai
        response = _model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return response.text.strip()
    except Exception as e:
        return f"[LLM Error: {e}]\n\n" + _fallback(prompt)


def _fallback(prompt: str) -> str:
    """Rule-based fallback when LLM is unavailable."""
    if "feature" in prompt.lower():
        return (
            "**Feature Engineering Notes (Auto-generated)**\n\n"
            "- Numerical columns have been standardised with StandardScaler.\n"
            "- Categorical columns have been one-hot encoded.\n"
            "- Missing values in numerical columns are imputed with the median.\n"
            "- Missing values in categorical columns are imputed with the mode.\n"
            "- No explicit domain knowledge was applied (LLM unavailable)."
        )
    if "report" in prompt.lower() or "explain" in prompt.lower():
        return (
            "**AutoML Report (Auto-generated)**\n\n"
            "The AutoML pipeline profiled your dataset, engineered features, "
            "and trained multiple models using 5-fold cross-validation. "
            "The best model was selected based on the highest cross-validation score. "
            "Please add a Gemini API key for a detailed narrative explanation."
        )
    return "LLM unavailable — add GEMINI_API_KEY to .env for AI-generated insights."
