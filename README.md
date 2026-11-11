# HarvestIQ – Intelligent Crop Recommendation System Using Machine Learning

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render Deploy](https://img.shields.io/badge/Render-Deploy%20Ready-46E3B7.svg?logo=render&logoColor=black)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **HarvestIQ** is an end-to-end, production-grade precision agriculture decision-support system. It leverages supervised machine learning to recommend the optimal crop for cultivation based on 7 essential agro-climatic soil and meteorological indicators.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Machine Learning Performance](#machine-learning-performance)
- [Technology Stack](#technology-stack)
- [Project Directory Structure](#project-directory-structure)
- [Agro-Climatic Parameters](#agro-climatic-parameters)
- [Local Installation & Setup](#local-installation--setup)
- [API Documentation](#api-documentation)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Deployment Guide (Render & GitHub)](#deployment-guide-render--github)
- [License](#license)

---

## Overview

Agricultural yield and profitability depend heavily on selecting crops aligned with local soil chemistry and meteorological conditions. Traditional farming choices often rely on heuristics or historical habit, resulting in suboptimal yields or excessive fertilizer application.

**HarvestIQ** bridges data science and precision agronomy by:
1. Validating 7 physiological soil and climate inputs against biological thresholds.
2. Ingesting calibrated parameters into an audited `RandomForestClassifier` machine learning pipeline.
3. Returning ranked primary recommendations along with top alternative candidates and uncalibrated model scores.
4. Delivering real-time recommendations through a responsive, modern web interface.

---

## Key Features

- **Empirically Audited Machine Learning Model**: Trained and evaluated across 5 candidate algorithms on 2,200 real agricultural records; achieves **99.55% test accuracy** and **100% top-3 accuracy**.
- **Ranked Crop Suggestions**: Outputs the primary recommended crop alongside 2 viable alternative crops with uncalibrated model scores.
- **Strict Boundary Validation**: Enforces strict biological boundaries for all 7 input parameters using Pydantic v2 schemas, returning structured HTTP 422 errors for out-of-range inputs.
- **Zero Mock / Zero Fake Policy**: All recommendations originate from the serialized Scikit-learn pipeline; fallback mechanisms cleanly report model state rather than producing fabricated data.
- **Production-Ready FastAPI Backend**: Asynchronous architecture, dynamic CORS origin management, lifespan state loading, and automated OpenAPI documentation (`/docs` and `/redoc`).
- **Responsive Dark-Themed Frontend**: Vanilla JavaScript UI featuring synchronized input sliders, quick agro-climatic preset buttons, real-time input validation, and live backend connection health diagnostics.
- **One-Click Cloud Deployment**: Pre-configured `render.yaml` Blueprint specification for deploying the FastAPI web service and static site on Render.

---

## System Architecture

```mermaid
flowchart TD
    User["Farmer / Agronomist"] -->|Adjusts parameters & submits| Frontend["HarvestIQ Web Client<br/>(HTML5 / CSS3 / Vanilla JS)"]
    Frontend -->|POST /recommend (JSON)| API["FastAPI Backend Service<br/>(Uvicorn / Python 3.11+)"]
    
    subgraph Validation ["Validation Layer"]
        API --> V1["Pydantic v2 Schema Validation"]
        V1 -->|Fails Bounds| E422["HTTP 422 Farmer-Friendly Error"]
        E422 --> Frontend
    end
    
    subgraph ML_Engine ["Inference Engine"]
        V1 -->|Valid Inputs| Pipeline["Serialized Pipeline<br/>(StandardScaler + RandomForest)"]
        Pipeline --> ModelArtifact["backend/models/harvestiq_model.joblib<br/>(3.58 MB)"]
        Pipeline --> Ranked["Predict Probabilities & Rank Classes"]
    end
    
    Ranked --> Response["HTTP 200 Recommendation Payload<br/>(Primary Crop + Top-3 Candidates)"]
    Response --> Frontend
    Frontend --> Render["Interactive Dashboard & Crop Diagnostics"]
```

---

## Machine Learning Performance

Five candidate algorithms were trained on an 80:20 stratified train-test split (1,760 train samples, 440 test samples) using 5-fold Stratified Cross-Validation on the 2,200-sample balanced Crop Recommendation dataset (22 distinct crop classes, 100 samples each).

| Algorithm | 5-Fold CV Accuracy | 5-Fold CV Macro F1 | Test Accuracy | Test Macro F1 | Top-3 Accuracy | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier (Baseline)** | 4.55% ± 0.00% | 0.40% ± 0.00% | 4.55% | 0.40% | 13.64% | No |
| **Logistic Regression** | 96.82% ± 0.66% | 96.78% ± 0.68% | 97.27% | 97.25% | 100.00% | No |
| **K-Nearest Neighbors (KNN)** | 96.53% ± 1.21% | 96.50% ± 1.24% | 97.95% | 97.93% | 99.77% | No |
| **Random Forest Classifier** | **99.26% ± 0.58%** | **99.26% ± 0.59%** | **99.55%** | **99.55%** | **100.00%** | **Yes (Champion)** |
| **Gradient Boosting Classifier** | 98.69% ± 0.34% | 98.69% ± 0.36% | 98.86% | 98.87% | 99.77% | No |

### Supported Crops (22 Classes)
`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.

---

## Technology Stack

- **Backend**: Python 3.11+, FastAPI 0.115+, Uvicorn 0.32+, Pydantic v2.9+
- **Machine Learning**: Scikit-Learn 1.5+, NumPy 1.26+, Pandas 2.2+, Joblib 1.4+
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism & Responsive CSS Variables), Vanilla JavaScript (ES6+ `fetch` API)
- **Quality Assurance**: Pytest 8.3+, HTTPX, Custom verification test suites (34/34 tests passing)
- **Deployment & Cloud**: Render (Web Service + Static Site), Git & GitHub

---

## Project Directory Structure

```text
trustvault/
├── .gitignore                      # Git exclusion rules (safely keeps model artifacts)
├── render.yaml                     # Render Blueprint specification (IaC)
├── README.md                       # Comprehensive project documentation
├── DEPLOYMENT_GUIDE.md             # Step-by-step GitHub & Render deployment instructions
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application, CORS, lifespan, exception handlers
│   │   ├── schemas.py              # Pydantic v2 request/response validation schemas
│   │   ├── services/
│   │   │   └── prediction.py       # Inference service, pipeline loader, top-3 ranker
│   │   └── utils/
│   │       └── validation.py       # Physiological boundaries and error formatting
│   ├── data/
│   │   ├── Crop_recommendation.csv # Verified original dataset (2,200 records, CC0)
│   │   └── processed/              # Cleaned dataset without duplicate/missing rows
│   ├── ml/
│   │   ├── train_model.py          # 5-algorithm comparative training pipeline
│   │   ├── evaluate_model.py       # Cross-validation & holdout evaluation script
│   │   └── preprocessing.py        # Feature scaler & train-test split pipeline
│   ├── models/
│   │   ├── harvestiq_model.joblib  # Serialized production pipeline (3.58 MB)
│   │   ├── scaler.joblib           # Standalone fitted StandardScaler
│   │   └── label_encoder.joblib    # Standalone fitted LabelEncoder
│   ├── reports/
│   │   ├── model_comparison.csv    # Benchmark comparison metrics across 5 models
│   │   ├── classification_report.csv # Per-crop precision, recall, and F1 scores
│   │   ├── confusion_matrix.png    # High-resolution 22x22 confusion matrix
│   │   └── evaluation_report.json  # Serialized test metrics & hyperparameter details
│   ├── tests/
│   │   ├── run_tests.py            # Automated backend unit test runner
│   │   ├── test_prompt5_api.py     # FastAPI endpoint and schema tests
│   │   ├── test_prompt6_frontend_integration.py # Full-stack integration audit
│   │   └── qa_backend_and_ml_suite.py # Comprehensive 34-case QA verification suite
│   ├── requirements.txt            # Locked Python production dependencies
│   ├── runtime.txt                 # Target Python runtime version (python-3.11.9)
│   └── .env.example                # Template for environment configuration
└── frontend/
    ├── index.html                  # Semantic, accessible UI layout
    ├── style.css                   # Responsive dark-theme styling & glassmorphism
    ├── app.js                      # UI logic, slider synchronization, async fetch handler
    └── config.js                   # Dynamic API base URL resolution & agro-climatic presets
```

---

## Agro-Climatic Parameters

| Parameter | Unit | Allowed Range | Default Value | Agronomic Role |
| :--- | :---: | :---: | :---: | :--- |
| **Nitrogen (N)** | kg/ha | `0` – `200` | `90` | Leaf biomass growth and chlorophyll synthesis |
| **Phosphorus (P)** | kg/ha | `0` – `200` | `42` | Root development, flowering, and seed viability |
| **Potassium (K)** | kg/ha | `0` – `300` | `43` | Drought tolerance, disease resistance, and water regulation |
| **Temperature** | °C | `0.0` – `50.0` | `20.8` | Ambient temperature during active crop growth season |
| **Humidity** | % | `0.0` – `100.0` | `82.0` | Canopy relative humidity and transpiration rate |
| **Soil pH** | pH | `3.0` – `10.0` | `6.5` | Soil acidity / alkalinity controlling nutrient availability |
| **Rainfall** | mm | `0.0` – `400.0` | `202.9` | Seasonal precipitation or total irrigation equivalent |

---

## Local Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git
- Web browser (Chrome, Firefox, Edge, Safari)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/harvestiq.git
cd harvestiq
```

### 2. Set Up Python Virtual Environment
```bash
# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the FastAPI Backend Server
```bash
# From within the backend/ directory
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API is now live at `http://127.0.0.1:8000`.
- Health Check: `http://127.0.0.1:8000/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 5. Launch the Frontend
The frontend requires no build steps. You can serve it using any HTTP server:
```bash
# From the project root in a separate terminal
python -m http.server 3000 --directory frontend
```
Open `http://localhost:3000` in your browser. The frontend will automatically detect the backend running at `http://127.0.0.1:8000`.

---

## API Documentation

### 1. `GET /health`
Verifies backend operational status and ML model availability.
```json
{
  "status": "healthy",
  "service": "HarvestIQ API",
  "version": "1.0.0",
  "model_loaded": true,
  "model_name": "RandomForestClassifier",
  "dataset_present": true,
  "python_version": "3.11.9"
}
```

### 2. `GET /model-info`
Provides detailed metadata on the active model, supported crops, and evaluation metrics.

### 3. `GET /parameters`
Returns official biological parameter ranges and validation bounds.

### 4. `POST /recommend`
Submits 7 agro-climatic values to generate ranked crop recommendations.

**Request Payload:**
```json
{
  "nitrogen": 90.0,
  "phosphorus": 42.0,
  "potassium": 43.0,
  "temperature": 20.87,
  "humidity": 82.00,
  "ph": 6.50,
  "rainfall": 202.93
}
```

**Success Response (`200 OK`):**
```json
{
  "success": true,
  "status": "success",
  "recommendation": {
    "primary_crop": "Rice",
    "confidence": 0.90,
    "top_candidates": [
      { "rank": 1, "crop": "Rice", "confidence": 0.90 },
      { "rank": 2, "crop": "Jute", "confidence": 0.10 },
      { "rank": 3, "crop": "Pomegranate", "confidence": 0.00 }
    ],
    "model_name": "RandomForestClassifier",
    "model_version": "1.0.0",
    "score_type": "uncalibrated model score"
  },
  "timestamp": "2026-11-11T17:30:00Z"
}
```

**Validation Error Response (`422 Unprocessable Entity`):**
```json
{
  "status": "error",
  "error_type": "validation_error",
  "message": "Input agro-climatic parameters violated biological boundary limits.",
  "errors": [
    {
      "parameter": "nitrogen",
      "value": 250.0,
      "message": "Value exceeds physiological threshold of 200.0 kg/ha."
    }
  ]
}
```

---

## Testing & Quality Assurance

HarvestIQ includes comprehensive automated test suites covering input schemas, biological validation rules, model inference accuracy, top-3 candidate extraction, and frontend API contracts.

```bash
# Run core verification tests
python backend/tests/run_tests.py

# Run frontend-backend integration tests
python backend/tests/test_prompt6_frontend_integration.py

# Run full QA test suite (34 test cases)
python backend/tests/qa_backend_and_ml_suite.py
```
All 34 test cases pass with a **100% success rate**. Detailed test logs are available in [`TESTING_REPORT.md`](file:///c:/Users/dasar/trustvault/TESTING_REPORT.md) and [`TEST_CASES.csv`](file:///c:/Users/dasar/trustvault/TEST_CASES.csv).

---

## Deployment Guide (Render & GitHub)

For complete, step-by-step instructions on deploying HarvestIQ to GitHub and Render, refer to the [`DEPLOYMENT_GUIDE.md`](file:///c:/Users/dasar/trustvault/DEPLOYMENT_GUIDE.md) document.

### Quick Overview:
1. **GitHub**: Commit project files and push to a new public or private repository. The serialized model artifact (`harvestiq_model.joblib`, 3.58 MB) is preserved directly in git without requiring Git LFS.
2. **Render Blueprint (Recommended)**: Connect your GitHub repository to Render and deploy using the included `render.yaml` specification.
3. **Manual Render Setup**:
   - **Backend Web Service**: Python runtime, root directory `backend`, start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check `/health`.
   - **Frontend Static Site**: Static runtime, root directory `frontend`, publish directory `.`.

---

## License

This project is licensed under the [MIT License](LICENSE).
The Crop Recommendation Dataset is available under the **Creative Commons CC0 1.0 Public Domain Dedication**.
