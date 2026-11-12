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


def classify_claim(text: str, threshold: float = 0.65) -> dict:
    if not text.strip():
        raise ValueError("Claim text must not be empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    result = load_classifier()(text, candidate_labels=LABELS, multi_label=False)
    confidence = float(result["scores"][0])
    return {
        "category": result["labels"][0],
        "confidence": confidence,
        "routing": "human_review" if confidence < threshold else "auto_triage",
        "model": "zero-shot transformer baseline",
    }
