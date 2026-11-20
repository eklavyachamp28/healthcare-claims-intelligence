"""Optional Hugging Face Transformers zero-shot baseline.

Downloads model weights on first use; CPU inference can be slow.
Synthetic/non-sensitive text only. No clinical or coverage decisions.
"""
from __future__ import annotations
from functools import lru_cache

LABELS = [
    "authorization", "eligibility", "coding",
    "documentation", "duplicate", "payment"
]


@lru_cache(maxsize=1)
def load_classifier(model_name: str = "facebook/bart-large-mnli"):
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=model_name, device=-1)


def classify_claim(text: str, threshold: float = 0.65, min_margin: float = 0.10) -> dict:
    if not text.strip():
        raise ValueError("Claim text must not be empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 <= min_margin <= 1:
        raise ValueError("min_margin must be between 0 and 1")
    result = load_classifier()(text, candidate_labels=LABELS, multi_label=False)
    confidence = float(result["scores"][0])
    margin = confidence - float(result["scores"][1])
    return {
        "score_margin": margin,
        "category": result["labels"][0],
        "confidence": confidence,
        "routing": "human_review" if confidence < threshold or margin < min_margin else "auto_triage",
        "model": "zero-shot transformer baseline",
    }
