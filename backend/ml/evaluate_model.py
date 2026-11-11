"""HarvestIQ Model Evaluation Script.

Evaluates an existing serialized model against holdout or full dataset,
outputting classification metrics and feature importances.
Does NOT fabricate metrics if a model has not been trained.
"""

from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def evaluate_existing_model(data_path: Path, models_dir: Path) -> bool:
    """Evaluate trained model performance against the dataset."""
    model_file = models_dir / "crop_recommendation_model.joblib"
    scaler_file = models_dir / "scaler.joblib"
    encoder_file = models_dir / "label_encoder.joblib"

    print("=" * 65)
    print(" HarvestIQ Model Evaluation")
    print("=" * 65)

    # Check for harvestiq_model.joblib or crop_recommendation_model.joblib
    harvestiq_file = models_dir / "harvestiq_model.joblib"
    if harvestiq_file.exists():
        model_file = harvestiq_file
    elif not model_file.exists():
        print(f"\n[EVALUATION ABORTED] Model artifact not found at: {model_file}")
        print("Please train a model first using: python ml/train_model.py or ml/train_and_evaluate.py")
        return False

    if not data_path.exists():
        print(f"\n[EVALUATION ABORTED] Evaluation dataset not found at: {data_path}")
        return False

    try:
        import joblib
        from sklearn.metrics import classification_report
    except ImportError as err:
        print(f"\n[DEPENDENCY ERROR] Missing required ML package: {err.name}")
        print("Please install requirements first: pip install -r requirements.txt")
        return False

    from ml.preprocessing import (
        EXPECTED_FEATURE_COLS,
        load_dataset,
        prepare_features_and_target,
        split_data,
    )

    print(f"\nLoading model from: {model_file}")
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file) if scaler_file.exists() else None
    encoder = joblib.load(encoder_file) if encoder_file.exists() else None

    print(f"Loading evaluation dataset from: {data_path}")
    df = load_dataset(data_path)

    is_pipeline = hasattr(model, "named_steps")
    if is_pipeline:
        # Pipeline encapsulates StandardScaler; pass raw features to prevent double-scaling
        X_raw = df[EXPECTED_FEATURE_COLS].values
        y_raw = df["label"].astype(str).str.strip().str.lower().values
        _, X_test, _, y_test = split_data(X_raw, y_raw, test_size=0.2, random_state=42)
    else:
        X_scaled, y_encoded, _, _ = prepare_features_and_target(
            df, scaler=scaler, encoder=encoder, fit_transforms=False
        )
        _, X_test, _, y_test = split_data(X_scaled, y_encoded, test_size=0.2, random_state=42)

    print("\nRunning inference on 20% stratified test set...")
    import pandas as pd
    import numpy as np
    X_test_df = pd.DataFrame(X_test, columns=EXPECTED_FEATURE_COLS)
    y_pred = model.predict(X_test_df)

    if isinstance(y_pred[0], str) and isinstance(y_test[0], (int, np.integer)):
        y_test_eval = encoder.inverse_transform(y_test)
        target_names = None
    elif isinstance(y_pred[0], (int, np.integer)) and isinstance(y_test[0], str):
        y_test_eval = encoder.transform(y_test)
        target_names = [str(c) for c in encoder.classes_] if encoder else None
    else:
        y_test_eval = y_test
        target_names = None

    report = classification_report(y_test_eval, y_pred, target_names=target_names, zero_division=0)
    print("\nDetailed Classification Report:")
    print("-" * 65)
    print(report)

    # Feature importances
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        print("\nFeature Importances:")
        print("-" * 65)
        importances = clf.feature_importances_
        for feat, imp in sorted(
            zip(EXPECTED_FEATURE_COLS, importances), key=lambda x: x[1], reverse=True
        ):
            bar = "#" * int(imp * 40)
            print(f"  {feat.upper():<12} | {imp * 100:5.2f}% | {bar}")

    return True


if __name__ == "__main__":
    data_path = backend_dir / "data" / "Crop_recommendation.csv"
    models_dir = backend_dir / "models"
    evaluate_existing_model(data_path, models_dir)
