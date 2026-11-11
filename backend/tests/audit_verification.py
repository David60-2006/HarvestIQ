"""Audit script for HarvestIQ ML Model verification."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def run_audit():
    model_path = Path("backend/models/harvestiq_model.joblib")
    if not model_path.exists():
        print("[FAIL] Model does not exist!")
        return

    model = joblib.load(model_path)
    print(f"1. Model Loaded: {model_path}")
    print(f"   Pipeline Steps: {list(model.named_steps.keys())}")
    scaler = model.named_steps["scaler"]
    rf = model.named_steps["classifier"]
    print(f"   Classifier Class: {rf.__class__.__name__}")
    print(f"   Fitted Trees (n_estimators): {len(rf.estimators_)}")
    print(f"   Classes count: {len(rf.classes_)}")
    print(f"   Classes: {list(rf.classes_)}")

    cols = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    # Test Samples from dataset
    test_cases = [
        {"name": "Rice Benchmark", "features": [90.0, 42.0, 43.0, 20.87, 82.0, 6.5, 202.93], "expected": "rice"},
        {"name": "Cotton Benchmark", "features": [118.0, 46.0, 19.0, 23.97, 81.42, 6.96, 69.84], "expected": "cotton"},
        {"name": "Coffee Benchmark", "features": [104.0, 18.0, 30.0, 23.60, 60.39, 6.78, 140.94], "expected": "coffee"},
        {"name": "Chickpea Benchmark", "features": [40.0, 72.0, 77.0, 17.02, 16.98, 7.48, 88.55], "expected": "chickpea"},
    ]

    print("\n2. Executing Real Predictions with Pipeline:")
    for tc in test_cases:
        df_in = pd.DataFrame([tc["features"]], columns=cols)
        pred = model.predict(df_in)[0]
        proba = model.predict_proba(df_in)[0]
        conf = float(np.max(proba) * 100)
        is_valid_crop = pred in rf.classes_
        print(f"   [{tc['name']}] Pred: '{pred}' | Match Expected: {pred == tc['expected']} | In 22 Classes: {is_valid_crop} | Confidence: {conf:.2f}%")

    # Item 10 Investigation: 90% confidence breakdown
    print("\n3. Investigating Confidence for Rice Benchmark [90, 42, 43, 20.87, 82.0, 6.5, 202.93]:")
    df_rice = pd.DataFrame([[90.0, 42.0, 43.0, 20.87, 82.0, 6.5, 202.93]], columns=cols)
    scaled_rice = scaler.transform(df_rice)
    
    # Check individual tree votes
    votes = []
    for i, tree in enumerate(rf.estimators_):
        tree_pred = tree.predict(scaled_rice)[0]
        # tree predict returns class index in tree.classes_
        class_name = rf.classes_[int(tree_pred)]
        votes.append(class_name)

    vote_counts = pd.Series(votes).value_counts().to_dict()
    print(f"   Individual Tree Votes across 100 estimators:")
    for crop, count in vote_counts.items():
        print(f"     - {crop}: {count} trees ({count / len(rf.estimators_) * 100:.1f}%)")

    proba_rice = model.predict_proba(df_rice)[0]
    top_indices = np.argsort(proba_rice)[::-1][:3]
    print(f"   Exact model.predict_proba() distribution:")
    for idx in top_indices:
        print(f"     - Class '{rf.classes_[idx]}': probability = {proba_rice[idx]:.4f} ({proba_rice[idx]*100:.2f}%)")

if __name__ == "__main__":
    run_audit()
