# Healthcare Claims Intelligence

A small NLP classification project that routes synthetic healthcare claim descriptions into operational categories and flags uncertain predictions for human review. It uses synthetic examples only; do not submit real patient or claims data.

## What it demonstrates
- Text preprocessing with word- and character-level TF-IDF features
- Logistic-regression multi-class classification
- Confidence-based review routing
- Reproducible train/test split and classification metrics
- FastAPI prediction endpoint

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs.

## Example
```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"description":"Provider submitted an outpatient imaging claim with missing authorization"}'
```

The training script saves `claims_model.joblib`; train before starting the API. This is a portfolio demonstration, not a clinical, coverage or payment decision system. Real deployment requires representative approved data, privacy/security review, bias and subgroup evaluation, human oversight, audit trails and validation with domain experts.

## Stack
Python · pandas · scikit-learn · joblib · FastAPI · pytest

## Transformer NLP extension (optional)
Install `pip install -r requirements-transformers.txt` to enable a **Hugging Face Transformers** zero-shot classification baseline using `facebook/bart-large-mnli`. Model weights are downloaded on first use; inference can be slow on CPU.

```python
from transformer_classifier import classify_claim
print(classify_claim("The claim was rejected due to missing prior authorization"))
```

The extension maps predictions to six operational categories and routes low-confidence results to human review. It is a separate optional baseline from the TF-IDF model, not a fine-tuned or clinically validated system. Never submit patient-identifiable data.
