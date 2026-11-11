"""HarvestIQ – Intelligent Crop Recommendation System FastAPI Application.

Provides RESTful endpoints for health monitoring, agro-climatic validation,
and machine learning-based crop recommendation.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    CropRecommendationRequest,
    CropRecommendationResponse,
    HealthResponse,
    ModelInfoResponse,
    ParameterBoundsResponse,
)
from app.services.prediction import prediction_service
from app.utils.validation import (
    PARAMETER_RANGES,
    format_pydantic_validation_error,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes services on startup and performs resource cleanup on shutdown.
    """
    # Startup: attempt to detect model and dataset
    prediction_service.load_model()
    data_file = Path(__file__).resolve().parent.parent / "data" / "Crop_recommendation.csv"

    print("=" * 60)
    print(" HarvestIQ – Intelligent Crop Recommendation API")
    print("=" * 60)
    print(f" Python Runtime: {sys.version.split()[0]}")
    print(f" Model Loaded:   {prediction_service.is_model_loaded()}")
    print(f" Dataset Found:  {data_file.exists()} ({data_file})")
    print("=" * 60)
    yield
    print("[HarvestIQ] Application shutdown complete.")


# Initialize FastAPI application with comprehensive OpenAPI metadata
app = FastAPI(
    title="HarvestIQ – Crop Recommendation API",
    description=(
        "Production-grade FastAPI service powering the HarvestIQ intelligent "
        "crop recommendation system. Validates 7 agro-climatic soil and weather "
        "parameters and computes optimal crop recommendations using trained "
        "Machine Learning models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# -------------------------------------------------------------------------
# Dynamic CORS Configuration (Production & Local Development)
# -------------------------------------------------------------------------
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env.strip():
    # Split comma-separated allowed origins provided via environment variable
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
else:
    # Default permissive origins for local development and universal fallbacks
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5500",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
        "*",  # Fallback wildcard for local testing
    ]

is_wildcard = "*" in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not is_wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Custom Exception Handler for Meaningful HTTP 422 Validation Errors
# -------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Catches Pydantic RequestValidationError and transforms it into
    a farmer-friendly, actionable HTTP 422 response specifying the exact
    parameter and physiological limits violated.
    """
    formatted_payload = format_pydantic_validation_error(exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=formatted_payload,
    )


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------
@app.get(
    "/",
    summary="API Root & Overview",
    tags=["General"],
)
async def root() -> Dict[str, Any]:
    """Provides service identification and interactive documentation pointers."""
    return {
        "service": "HarvestIQ – Intelligent Crop Recommendation API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "parameters": "/parameters",
            "recommend": "/recommend",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Readiness Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Performs a real-time system health evaluation.
    Reports operational status, model readiness, model name, dataset availability,
    and active Python runtime version.
    """
    data_file = Path(__file__).resolve().parent.parent / "data" / "Crop_recommendation.csv"
    model_loaded = prediction_service.is_model_loaded()

    return HealthResponse(
        status="healthy",
        service="HarvestIQ API",
        version="1.0.0",
        model_loaded=model_loaded,
        model_name=prediction_service.model_name if model_loaded else None,
        dataset_present=data_file.exists(),
        python_version=sys.version.split()[0],
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Inspect Active ML Model Metadata & Metrics",
    tags=["Metadata"],
)
async def get_model_info() -> ModelInfoResponse:
    """
    Return model algorithm name, version, class labels, pipeline structure,
    and empirical validation evaluation metrics.
    """
    return prediction_service.get_model_info()


@app.get(
    "/parameters",
    response_model=ParameterBoundsResponse,
    summary="Get Agro-Climatic Parameter Boundaries",
    tags=["Metadata"],
)
async def get_parameters() -> ParameterBoundsResponse:
    """
    Returns the official physiological bounds and units for all 7 input parameters.
    Used dynamically by frontends to configure form validation sliders and limits.
    """
    return ParameterBoundsResponse(parameters=PARAMETER_RANGES)  # type: ignore


@app.post(
    "/recommend",
    response_model=CropRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Crop Recommendation",
    tags=["Prediction"],
    responses={
        200: {
            "description": "Successful crop recommendation with ranked candidates and uncalibrated model scores",
        },
        422: {
            "description": "Validation Error – One or more parameters outside allowed boundaries",
        },
        503: {
            "description": "Service Unavailable – Machine learning model artifact not loaded",
        },
    },
)
async def recommend_crop(
    payload: CropRecommendationRequest,
) -> JSONResponse:
    """
    Evaluate 7 agro-climatic parameters and predict the most suitable crop using the saved ML pipeline.

    - **Nitrogen (N)**: 0 – 200 kg/ha
    - **Phosphorus (P)**: 0 – 200 kg/ha
    - **Potassium (K)**: 0 – 300 kg/ha
    - **Temperature**: 0 – 50 °C
    - **Humidity**: 0 – 100 %
    - **Soil pH**: 3.0 – 10.0
    - **Rainfall**: 0 – 400 mm

    Returns primary crop recommendation, top-3 ranked alternatives, uncalibrated scores,
    and model version metadata.
    """
    response = prediction_service.predict(payload)
    if not response.success:
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if response.status == "model_unavailable"
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        return JSONResponse(status_code=status_code, content=response.model_dump())

    return JSONResponse(status_code=status.HTTP_200_OK, content=response.model_dump())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[HarvestIQ] Starting server on {host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
