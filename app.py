"""FastAPI service for claims-text triage."""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

MODEL_PATH = Path("claims_model.joblib")
REVIEW_THRESHOLD = 0.55
app = FastAPI(title="Healthcare Claims Intelligence API", version="1.0.0")

class Claim(BaseModel):
    description: str = Field(min_length=8, max_length=5000)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/classify")
def classify(claim: Claim):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Run python train.py first.")
    model = load(MODEL_PATH)
    probabilities = model.predict_proba(pd.Series([claim.description]))[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    return {"category": str(model.classes_[index]),
            "confidence": confidence,
            "route": "auto_triage" if confidence >= REVIEW_THRESHOLD else "human_review",
            "review_threshold": REVIEW_THRESHOLD}
