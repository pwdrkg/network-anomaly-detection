"""
genai_explain.py — OPTIONAL GenAI-enhanced feature (Step 9).

Turns a model prediction into a short, analyst-friendly incident summary using
an LLM. This augments the numeric output of the detector with a plain-language
triage note, so an analyst can act faster.

The detector itself is a plain, auditable Random Forest — the LLM is used only
to *explain* a decision, never to make it. Requires an Anthropic API key:

    export ANTHROPIC_API_KEY=sk-ant-...
    python src/genai_explain.py

If the key or SDK is missing, the module degrades gracefully to a deterministic
template so the pipeline never hard-depends on an external service.
"""
import os
from typing import Optional


def build_prompt(record: dict, proba: float, top_features: list[tuple[str, float]]) -> str:
    feats = "\n".join(f"  - {k}: {v}" for k, v in top_features)
    return (
        "You are a SOC (security operations) assistant. In 2-3 sentences, write a "
        "concise, non-alarmist triage note for the network connection below. "
        "State the likely verdict, the top contributing factors in plain English, "
        "and one suggested next step. Do not invent details beyond what is given.\n\n"
        f"Attack probability: {proba:.1%}\n"
        f"Key connection features:\n{feats}\n"
    )


def explain(record: dict, proba: float, top_features: list[tuple[str, float]],
            model: str = "claude-sonnet-4-5") -> str:
    """Return an LLM-written triage note, or a template fallback."""
    prompt = build_prompt(record, proba, top_features)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:  # noqa: BLE001 — never let the LLM break scoring
            return _fallback(proba, top_features) + f"  (LLM unavailable: {e})"
    return _fallback(proba, top_features)


def _fallback(proba: float, top_features: list[tuple[str, float]]) -> str:
    verdict = "likely ATTACK" if proba >= 0.5 else "likely NORMAL"
    drivers = ", ".join(k for k, _ in top_features[:3])
    step = ("Escalate for investigation and check the source host."
            if proba >= 0.5 else "No action needed; log for baseline.")
    return (f"Connection scored {proba:.0%} — {verdict}. "
            f"Main contributing factors: {drivers}. {step}")


if __name__ == "__main__":
    # illustrative example (uses a real attack-like record)
    demo_record = {"proto": "sctp", "sttl": 254, "sload": 46222220, "sbytes": 104}
    demo_top = [("sttl", 0.31), ("ttl_diff", 0.18), ("ct_state_ttl", 0.12)]
    print(explain(demo_record, proba=0.98, top_features=demo_top))
