"""HarvestIQ Machine Learning Prediction Service.

Loads serialized model artifacts, runs real inference, formats ranked recommendation responses,
and provides model inspection metadata.
Strict Rule: Does NOT fake predictions or claim accuracy if a model has not been trained.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas import (
    CropAlternative,
    CropRecommendationRequest,
    CropRecommendationResponse,
    ModelInfoResponse,
    RecommendationItem,
)


class PredictionService:
    """Service managing ML model lifecycle, inference, and metadata for crop recommendation."""

    def __init__(self, models_dir: Optional[Path] = None):
        # Default models directory is backend/models
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

        self.harvestiq_model_path = self.models_dir / "harvestiq_model.joblib"
        self.compat_model_path = self.models_dir / "crop_recommendation_model.joblib"
        self.scaler_path = self.models_dir / "scaler.joblib"
        self.encoder_path = self.models_dir / "label_encoder.joblib"

        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.is_ready: bool = False

        self.model_name: str = "None"
        self.model_version: str = "1.0.0"
        self.score_type: str = "uncalibrated model score"
        self.supported_crops: List[str] = []
        self.pipeline_steps: List[str] = []
        self.evaluation_metrics: Optional[Dict[str, Any]] = None

        # Load model once during initialization
        self.load_model()

    def load_model(self) -> bool:
        """
        Load trained model, scaler, and label encoder from disk if present.
        Supports both unified sklearn Pipeline and standalone models.
        Returns True if model was successfully loaded, False otherwise.
        """
        target_model_file = None
        if self.harvestiq_model_path.exists():
            target_model_file = self.harvestiq_model_path
        elif self.compat_model_path.exists():
            target_model_file = self.compat_model_path

        if target_model_file is None:
            self.is_ready = False
            self.model = None
            self.model_name = "None"
            self.supported_crops = []
            self.pipeline_steps = []
            return False

        try:
            import joblib

            self.model = joblib.load(target_model_file)

            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.scaler = None

            if self.encoder_path.exists():
                self.label_encoder = joblib.load(self.encoder_path)
            else:
                self.label_encoder = None

            # Detect pipeline steps and model name
            if hasattr(self.model, "named_steps"):
                self.pipeline_steps = list(self.model.named_steps.keys())
                clf = self.model.named_steps.get("classifier", self.model.steps[-1][1])
                self.model_name = clf.__class__.__name__
            else:
                self.pipeline_steps = []
                self.model_name = self.model.__class__.__name__

            # Detect supported crop classes
            if hasattr(self.model, "classes_"):
                self.supported_crops = [str(c) for c in self.model.classes_]
            elif self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
                self.supported_crops = [str(c) for c in self.label_encoder.classes_]
            else:
                self.supported_crops = []

            # Load actual training evaluation metrics if available
            eval_report_file = self.models_dir.parent / "reports" / "evaluation_report.json"
            if eval_report_file.exists():
                try:
                    with open(eval_report_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.evaluation_metrics = data.get("selected_model_metrics")
                except Exception:
                    self.evaluation_metrics = None
            else:
                self.evaluation_metrics = None

            self.is_ready = True
            return True
        except Exception as e:
            print(f"[PredictionService] Warning: Failed to load model artifact: {e}")
            self.is_ready = False
            self.model = None
            self.model_name = "None"
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for live inference."""
        return self.is_ready and self.model is not None

    def get_model_info(self) -> ModelInfoResponse:
        """Return model metadata, version, classes, and actual training evaluation metrics."""
        return ModelInfoResponse(
            model_name=self.model_name if self.is_model_loaded() else "None",
            model_version=self.model_version,
            is_loaded=self.is_model_loaded(),
            score_type=self.score_type,
            pipeline_steps=self.pipeline_steps,
            supported_crops=self.supported_crops,
            classes_count=len(self.supported_crops),
            features=["n", "p", "k", "temperature", "humidity", "ph", "rainfall"],
            features_count=7,
            evaluation_metrics=self.evaluation_metrics,
        )

    def predict(
        self, request: CropRecommendationRequest
    ) -> CropRecommendationResponse:
        """
        Execute prediction pipeline for validated crop parameters.

        If model is loaded: Runs real ML inference and returns predicted crop,
        top-three ranked recommendations, uncalibrated scores, model metadata, and timestamp.
        If model is NOT loaded: Returns failure status 'model_unavailable'.
        DOES NOT fake predictions or fake accuracy.
        """
        validated_input = request.to_dict()

        if not self.is_model_loaded():
            return CropRecommendationResponse(
                success=False,
                status="model_unavailable",
                model_status="not_trained",
                recommended_crop=None,
                recommendations=[],
                model_name="None",
                model_version=self.model_version,
                score_type=self.score_type,
                confidence=None,
                alternatives=None,
                validated_input=validated_input,
                message=(
                    "Machine learning model artifact is not available. Please ensure the verified "
                    "model 'harvestiq_model.joblib' is present in backend/models/."
                ),
            )

        try:
            import numpy as np
            import pandas as pd

            # 1. Feature extraction in model expected order: [N, P, K, Temp, Humidity, pH, Rainfall]
            feature_names = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
            features_df = pd.DataFrame([request.to_feature_list()], columns=feature_names)
            raw_features = features_df.values

            # 2. Scale features if using standalone model; Pipelines scale internally
            is_pipeline = hasattr(self.model, "named_steps")
            if is_pipeline:
                features = features_df
            elif self.scaler is not None:
                features = self.scaler.transform(raw_features)
            else:
                features = raw_features

            # 3. Model inference
            predicted_class_raw = self.model.predict(features)[0]

            # 4. Decode label (handles both direct string models and integer-encoded models)
            if isinstance(predicted_class_raw, (int, np.integer)):
                if self.label_encoder is not None:
                    recommended_crop = str(self.label_encoder.inverse_transform([predicted_class_raw])[0])
                elif hasattr(self.model, "classes_") and len(self.model.classes_) > predicted_class_raw:
                    recommended_crop = str(self.model.classes_[predicted_class_raw])
                else:
                    recommended_crop = str(predicted_class_raw)
            else:
                recommended_crop = str(predicted_class_raw)

            # 5. Calculate ranked recommendations and uncalibrated scores
            recommendations: List[RecommendationItem] = []
            alternatives: List[CropAlternative] = []
            confidence: Optional[float] = None

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                top_indices = np.argsort(probabilities)[::-1]

                # Top-3 recommendations ranked using actual model probability scores
                for rank_idx, idx in enumerate(top_indices[:3], start=1):
                    prob = float(probabilities[idx])
                    if hasattr(self.model, "classes_") and isinstance(self.model.classes_[0], str):
                        crop_name = str(self.model.classes_[idx])
                    elif self.label_encoder is not None:
                        crop_name = str(self.label_encoder.inverse_transform([idx])[0])
                    else:
                        crop_name = str(idx)

                    recommendations.append(
                        RecommendationItem(
                            crop=crop_name.capitalize(),
                            score=round(prob, 4),
                            confidence_percentage=round(prob * 100.0, 2),
                            rank=rank_idx,
                        )
                    )

                # Primary prediction confidence
                primary_prob = float(probabilities[top_indices[0]])
                confidence = round(primary_prob * 100.0, 2)

                # Secondary alternative candidates (ranks 2 and 3)
                for item in recommendations[1:]:
                    if item.score > 0.005:  # At least 0.5% score
                        alternatives.append(
                            CropAlternative(
                                crop=item.crop.lower(),
                                confidence_percentage=item.confidence_percentage,
                            )
                        )

            return CropRecommendationResponse(
                success=True,
                status="success",
                model_status="ready",
                recommended_crop=recommended_crop.capitalize(),
                recommendations=recommendations,
                model_name=self.model_name,
                model_version=self.model_version,
                score_type=self.score_type,
                confidence=confidence,
                alternatives=alternatives if alternatives else None,
                validated_input=validated_input,
                message=f"Optimal crop recommended based on agro-climatic profile: {recommended_crop.capitalize()}.",
            )

        except Exception as e:
            return CropRecommendationResponse(
                success=False,
                status="error",
                model_status="ready",
                recommended_crop=None,
                recommendations=[],
                model_name=self.model_name,
                model_version=self.model_version,
                score_type=self.score_type,
                confidence=None,
                alternatives=None,
                validated_input=validated_input,
                message=f"Inference error during model prediction: {str(e)}",
            )


# Global singleton instance for application use
prediction_service = PredictionService()
