"""HarvestIQ Machine Learning Training Pipeline.

Trains a Random Forest classifier on verified agro-climatic data.
Strict Rule: Verifies the dataset before training. Will NOT generate fake models
or fake accuracy metrics if the verified dataset is absent.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def train_pipeline(
    data_path: Path,
    output_dir: Path,
    n_estimators: int = 100,
    random_state: int = 42,
) -> bool:
    """
    Executes the end-to-end model training workflow.

    Returns True if training succeeded, False if dataset was missing or invalid.
    """
    print("=" * 65)
    print(" HarvestIQ Machine Learning Training Pipeline")
    print("=" * 65)

    # 1. Dataset Verification Check (Executed before heavy imports)
    if not data_path.exists():
        print(f"\n[DATASET VERIFICATION FAILED]")
        print(f"No dataset file found at: {data_path}")
        print("\nRequired Steps:")
        print("1. Obtain the verified Crop Recommendation benchmark CSV dataset.")
        print(f"2. Save the file to: {data_path}")
        print("3. Ensure the CSV contains the 7 features (N, P, K, temperature,")
        print("   humidity, ph, rainfall) and target label ('label').")
        print("4. Install backend dependencies: pip install -r requirements.txt")
        print("5. Re-run this script: python ml/train_model.py\n")
        print("[Status] Training aborted safely. No unverified model generated.")
        return False

    # Check dependencies before proceeding
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import joblib
    except ImportError as err:
        print(f"\n[DEPENDENCY ERROR] Missing required ML package: {err.name}")
        print("Please install requirements first: pip install -r requirements.txt")
        return False

    from ml.preprocessing import (
        load_dataset,
        prepare_features_and_target,
        split_data,
    )

    print(f"\n[Step 1/5] Loading verified dataset from: {data_path}")
    try:
        df = load_dataset(data_path)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return False

    num_samples = len(df)
    unique_crops = df["label"].nunique()
    print(f"   -> Samples: {num_samples}")
    print(f"   -> Unique crop classes: {unique_crops}")

    # 2. Preprocessing & Feature Extraction
    print("\n[Step 2/5] Preprocessing features and encoding labels...")
    X_scaled, y_encoded, scaler, encoder = prepare_features_and_target(df)

    # 3. Stratified Train-Test Split
    print("\n[Step 3/5] Performing stratified 80/20 train-test split...")
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y_encoded, test_size=0.2, random_state=random_state
    )
    print(f"   -> Training samples: {len(X_train)}")
    print(f"   -> Testing samples:  {len(X_test)}")

    # 4. Model Training
    print(f"\n[Step 4/5] Training Random Forest Classifier (trees={n_estimators})...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import joblib

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. Real Model Evaluation (no fake metrics)
    print("\n[Step 5/5] Evaluating model performance on holdout test set...")
    y_pred = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    test_recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print(f"   -> Real Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"   -> Weighted Precision:  {test_precision * 100:.2f}%")
    print(f"   -> Weighted Recall:     {test_recall * 100:.2f}%")
    print(f"   -> Weighted F1 Score:   {test_f1 * 100:.2f}%")

    # Serialize Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    harvestiq_file = output_dir / "harvestiq_model.joblib"
    model_file = output_dir / "crop_recommendation_model.joblib"
    scaler_file = output_dir / "scaler.joblib"
    encoder_file = output_dir / "label_encoder.joblib"
    meta_file = output_dir / "model_metadata.json"

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("scaler", scaler), ("classifier", model)])

    print(f"\nSaving model artifacts to: {output_dir}")
    joblib.dump(pipeline, harvestiq_file)
    joblib.dump(pipeline, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(encoder, encoder_file)

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": num_samples,
        "classes_count": unique_crops,
        "classes": [str(c) for c in encoder.classes_],
        "metrics": {
            "test_accuracy": test_acc,
            "weighted_precision": test_precision,
            "weighted_recall": test_recall,
            "weighted_f1": test_f1,
        },
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"   -> Saved model:     {model_file.name}")
    print(f"   -> Saved scaler:    {scaler_file.name}")
    print(f"   -> Saved encoder:   {encoder_file.name}")
    print(f"   -> Saved metadata:  {meta_file.name}")
    print("\n[SUCCESS] HarvestIQ model training complete and verified.")
    return True


if __name__ == "__main__":
    data_dir = backend_dir / "data"
    data_path = data_dir / "Crop_recommendation.csv"
    output_dir = backend_dir / "models"

    train_pipeline(data_path=data_path, output_dir=output_dir)
