from train import build_pipeline, make_dataset

def test_dataset_contains_expected_claim_categories():
    data = make_dataset()
    assert data["category"].nunique() == 6
    assert data["description"].str.len().min() > 0

def test_classifier_predicts_known_training_examples():
    data = make_dataset()
    model = build_pipeline().fit(data["description"], data["category"])
    prediction = model.predict(["missing prior authorization for imaging claim"])
    assert prediction[0] in set(data["category"])
