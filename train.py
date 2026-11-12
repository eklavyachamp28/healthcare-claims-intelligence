"""Train a synthetic healthcare-claims text classifier."""
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

EXAMPLES = [
 ("authorization", "claim denied due to missing prior authorization"),
 ("authorization", "pre approval number is not present for imaging service"),
 ("authorization", "service requires referral and authorization from payer"),
 ("eligibility", "member coverage inactive on date of service"),
 ("eligibility", "patient insurance eligibility could not be verified"),
 ("eligibility", "subscriber policy terminated before treatment date"),
 ("coding", "diagnosis code does not support the billed procedure"),
 ("coding", "incorrect procedure code submitted on claim form"),
 ("coding", "modifier missing from outpatient surgery billing"),
 ("documentation", "medical records required to support billed service"),
 ("documentation", "clinical notes and discharge summary are missing"),
 ("documentation", "request additional documentation from provider"),
 ("duplicate", "same claim submitted twice for identical date and service"),
 ("duplicate", "duplicate invoice detected for this member encounter"),
 ("duplicate", "previously paid claim appears to have been resubmitted"),
 ("payment", "claim approved and payment released to provider"),
 ("payment", "reimbursement amount calculated after adjudication"),
 ("payment", "claim processed successfully with allowed amount"),
]

def make_dataset():
    rows = []
    for label, text in EXAMPLES:
        rows.extend([(label, text), (label, text + " for member claim"),
                     (label, "review claim: " + text)])
    return pd.DataFrame(rows, columns=["category", "description"])

def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
    ])
    return Pipeline([("features", features),
                     ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))])

def train(output_path="claims_model.joblib"):
    data = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["description"], data["category"], test_size=0.25,
        random_state=42, stratify=data["category"])
    model = build_pipeline().fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    dump(model, output_path)
    return model

if __name__ == "__main__":
    train()
