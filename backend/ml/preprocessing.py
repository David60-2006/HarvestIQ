"""HarvestIQ ML Preprocessing & Feature Engineering Module.

Provides functions for loading, validating, cleaning, scaling, and splitting
the agro-climatic crop recommendation dataset.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


EXPECTED_FEATURE_COLS: List[str] = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET_COL: str = "label"


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame contains required agro-climatic columns.
    Matches columns case-insensitively.
    """
    cols = [c.strip().lower() for c in df.columns]
    missing: List[str] = []

    for expected in EXPECTED_FEATURE_COLS:
        if expected not in cols:
            missing.append(expected)

    if TARGET_COL not in cols:
        missing.append(TARGET_COL)

    return len(missing) == 0, missing


def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load crop recommendation CSV dataset and perform initial schema checks.
    Raises FileNotFoundError if file is missing, or ValueError if schema is invalid.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please place Crop_recommendation.csv into backend/data/."
        )

    df = pd.read_csv(filepath)

    # Standardize column names to lower case
    df.columns = [c.strip().lower() for c in df.columns]

    # Schema validation
    is_valid, missing = validate_dataset_schema(df)
    if not is_valid:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Expected features: {EXPECTED_FEATURE_COLS} and target: '{TARGET_COL}'."
        )

    # Clean nulls or duplicates if any
    initial_len = len(df)
    df = df.dropna().drop_duplicates()
    cleaned_len = len(df)
    if cleaned_len < initial_len:
        print(f"[Preprocessing] Removed {initial_len - cleaned_len} duplicate or null records.")

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[LabelEncoder] = None,
    fit_transforms: bool = True,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Extracts features and target, scales features, and encodes crop labels.

    Args:
        df: Input DataFrame
        scaler: Pre-existing StandardScaler (or None to fit a new one)
        encoder: Pre-existing LabelEncoder (or None to fit a new one)
        fit_transforms: Whether to fit scaler and encoder on this data

    Returns:
        (X_scaled, y_encoded, fitted_scaler, fitted_encoder)
    """
    X = df[EXPECTED_FEATURE_COLS].values.astype(np.float64)
    y_raw = df[TARGET_COL].astype(str).str.strip().str.lower().values

    # Feature scaling
    if fit_transforms or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Target label encoding
    if fit_transforms or encoder is None:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_raw)
    else:
        y_encoded = encoder.transform(y_raw)

    return X_scaled, y_encoded, scaler, encoder


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split to preserve crop class proportions across train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
