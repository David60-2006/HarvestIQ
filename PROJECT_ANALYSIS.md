# HARVESTIQ – Project Inspection & Technical Analysis Report

**Project Name:** HARVESTIQ – An Intelligent Crop Recommendation System Using Machine Learning  
**Workspace Inspected:** `c:\Users\dasar\trustvault`  
**Analysis Date:** November 2026  
**Architect:** Senior Full-Stack Developer & Machine Learning Architect  

---

## Executive Summary

A comprehensive, non-destructive audit of the workspace located at `c:\Users\dasar\trustvault` was performed.

### Key Audit Findings:
1. **Workspace Identity Discrepancy:** The existing workspace currently houses **"TrustVault"**—a Zero-Trust enterprise document security and cryptographic proof-chain verification platform built with React 19, TypeScript, Vite, Supabase, and Node.js/Express.
2. **Crop Recommendation Logic:** **Zero crop recommendation logic exists anywhere in the repository.** There are no agro-climatic inputs (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, or Rainfall), no hardcoded crop lookup tables, and no agricultural business rules.
3. **Machine Learning Model:** **No trained machine learning model exists.** A scan for `.pkl`, `.joblib`, `.h5`, `.pt`, `.onnx`, or `.model` files yielded zero results.
4. **Python Backend & Dataset:** **No Python backend and no agricultural dataset exist in the project.** A search for `.py`, `.ipynb`, or `.csv` files returned zero files in the project. (Note: Python 3.13.15 is installed on the host operating system at `C:\Users\dasar\AppData\Local\Programs\Python\Python313\python.exe`, but has not been initialized or integrated into this workspace).
5. **Existing Files Status:** All existing files have been strictly preserved without any modification or deletion.

---

## 1. Deep-Dive Inspection of Existing HTML, CSS, and JavaScript/TypeScript Files

### 1.1 Directory Tree Analysis
```
c:\Users\dasar\trustvault\
├── backend/                   # Node.js + Express TypeScript backend
│   ├── src/server.ts          # Minimal Express server (health endpoints)
│   ├── dist/server.js         # Compiled JavaScript output
│   ├── .env                   # Supabase credentials & PORT
│   └── package.json           # Express, Supabase JS, tsx, typescript
├── client/                    # Unmodified Vite + TypeScript boilerplate
│   ├── index.html             # Basic template (<div id="app">)
│   └── src/
│       ├── main.ts            # Vite getting-started demo with counter
│       ├── counter.ts         # Sample counter setup function
│       └── style.css          # Vite boilerplate stylesheet
├── frontend/                  # React 19 + TypeScript + Vite project
│   ├── index.html             # Document titled "trustvault-web"
│   ├── package.json           # React 19, Lucide icons, Supabase JS, Vite
│   └── src/                   # React components, styles, types, and utils
├── src/                       # Exact duplicate clone of frontend/src/
│   ├── components/            # 12 React components (TrustVault UI)
│   ├── types/index.ts         # TypeScript interfaces for security platform
│   ├── utils/crypto.ts        # Web Crypto SHA-256 calculation
│   ├── supabase.ts            # Supabase client initialization
│   ├── App.tsx                # Main application component with tab routing
│   ├── App.css                # 4,685-line enterprise dark-mode stylesheet
│   └── index.css              # Reset & typography styles
├── index.html                 # Root index.html (<div id="root">)
├── package.json               # Root Vite/React configuration
└── vite.config.ts             # Vite configuration with React plugin
```

### 1.2 HTML Files
- **`c:\Users\dasar\trustvault\index.html` (and `frontend/index.html`):**
  - Title: `<title>trustvault-web</title>`
  - Viewport meta tag configured.
  - Mounts `<div id="root"></div>` and executes `<script type="module" src="/src/main.tsx"></script>`.
- **`c:\Users\dasar\trustvault\client\index.html`:**
  - Title: `<title>client</title>`
  - Mounts `<div id="app"></div>` and runs `<script type="module" src="/src/main.ts"></script>`.

### 1.3 CSS Files
- **`src/App.css` (4,685 lines):**
  - **Design System:** Highly refined enterprise dark-mode system.
  - **Color Tokens:**
    - Primary App Background: `--bg-app: #070c16;`
    - Surfaces & Cards: `--bg-surface: #0c1424;`, `--bg-card: #101c33;`
    - Accent Greens: `--emerald-main: #10b981;`, `--emerald-glow: rgba(16, 185, 129, 0.18);`
    - Supporting Accents: `--cyan-main: #06b6d4;`, `--blue-main: #3b82f6;`, `--amber-main: #f59e0b;`
  - **Typography:** DM Sans, Space Grotesk (headings), and JetBrains Mono (cryptographic hashes and code).
  - **Layout Patterns:** Glassmorphism, CSS grid and flex layouts, modal overlays, animated status badges, custom scrollbars, and responsive sidebars.
  - *Architectural Recommendation:* This emerald/dark-slate color palette is aesthetically ideal for an advanced agricultural intelligence dashboard (HarvestIQ).
- **`src/index.css`:**
  - Minimal CSS reset and root styling.
- **`client/src/style.css`:**
  - Standard Vite starter styling.

### 1.4 JavaScript / TypeScript Files
- **`src/App.tsx` (922 lines):**
  - Implements the complete client-side application logic for TrustVault.
  - State management for active navigation tabs (`dashboard`, `documents`, `trustlens`, `proofchain`, `members`, `settings`).
  - Contains mock genesis blocks (`initialProofChainBlocks`) and zero-trust decision logs (`initialDecisions`).
- **`src/components/` Components:**
  - `LandingPage.tsx`: Public hero section, feature cards, zero-trust architecture breakdown, security metrics, and authentication triggers.
  - `Navbar.tsx` & `Sidebar.tsx`: Navigation bars with active workspace selectors, search buttons, and tab switches.
  - `DashboardOverview.tsx`: Stat widgets, recent activity feed, security health scores.
  - `DocumentExplorer.tsx`: Document table, file upload interface with SHA-256 client hashing.
  - `ProofChainView.tsx`: Cryptographic ledger visualizing block hashes.
  - `TrustLensView.tsx`: Explainable security decision breakdown.
  - `TeamMembersView.tsx`: Workspace role management.
  - `SettingsProfileView.tsx`: System policies and 2FA toggles.
  - `AuthModal.tsx`, `CreateWorkspaceModal.tsx`, `SearchModal.tsx`: Overlay modals.
- **`src/utils/crypto.ts`:**
  - Calculates browser-native SHA-256 checksums using `window.crypto.subtle.digest`.
- **`backend/src/server.ts`:**
  - Express.js server on port 5000 with CORS and JSON body-parsing.
  - Exposes `GET /` and `GET /health`.

---

## 2. Current Website Design, Layout, Sections, and Features

The current frontend website presents an enterprise cybersecurity platform with two modes:
1. **Unauthenticated Public Landing Page (`LandingPage.tsx`):**
   - **Header / Brand:** TrustVault Enterprise Security mark with a shield icon.
   - **Hero Section:** Headline: "Zero-Trust Document Security & Real-Time Access Governance", CTA buttons for "Get Enterprise Access" and "Live Demo".
   - **Core Pillars:** Cards for Zero-Trust Access, TrustLens Decisions, and ProofChain Ledger.
   - **Enterprise Proof Metrics:** Grid displaying SHA-256 / AES-256, Multi-Tenant RLS, Immediate Alerting, 99.99% Availability.
   - **Workflow Timeline:** 4-step workflow (Upload & Hash, Zero-Trust Evaluation, Explainable Decision, Immutable Logging).
2. **Authenticated Dashboard Console (`App.tsx`):**
   - **Sidebar Navigation:** Collapsible navigation with workspace switcher and user profile.
   - **Dashboard Overview:** Key metric counters, quick actions, and recent audit activity.
   - **Document Explorer:** Upload drag-and-drop zone, file classification badges (Confidential, Restricted, Internal), and SHA-256 verification tags.
   - **ProofChain View:** Blockchain-inspired visual chain showing parent-child block linkages.
   - **TrustLens View:** Decision audit inspector with risk scoring dials (0–100).

---

## 3. Existing Functionality & Bugs Identified

### 3.1 Structural Redundancy & Confusion
- **Identical Code Trees:** The directories `c:\Users\dasar\trustvault\src\` and `c:\Users\dasar\trustvault\frontend\src\` are 100% duplicates. Running builds from the root vs. `frontend/` can create conflicting or desynchronized build artifacts.
- **Orphan Boilerplate:** The `client/` folder is an unused default Vite starter project that adds clutter without providing any application functionality.

### 3.2 Backend-Frontend Disconnection
- The Node/Express backend (`backend/src/server.ts`) is completely detached from the frontend. The frontend performs direct client-side database queries via Supabase JS (`src/supabase.ts`), leaving the Node server largely dormant.

### 3.3 Missing Environment Configurations
- `src/supabase.ts` uses `import.meta.env.VITE_SUPABASE_URL` and `import.meta.env.VITE_SUPABASE_ANON_KEY`. If `.env` or `.env.local` is missing or unpopulated, initialization falls back to empty strings, causing silent API failures on authentication and data fetching.

### 3.4 Hardcoded State
- Cryptographic blocks, access decisions, and member lists are initialized from in-memory arrays in `App.tsx` rather than being fetched from a persistent backend API.

---

## 4. Crop Recommendation Logic Analysis

| Aspect | Current Status in Codebase |
| :--- | :--- |
| **Nitrogen (N) Input** | ❌ Not present |
| **Phosphorus (P) Input** | ❌ Not present |
| **Potassium (K) Input** | ❌ Not present |
| **Temperature (°C) Input** | ❌ Not present |
| **Humidity (%) Input** | ❌ Not present |
| **Soil pH (0-14) Input** | ❌ Not present |
| **Rainfall (mm) Input** | ❌ Not present |
| **Crop Recommendation Rules** | ❌ Not present (None hardcoded, none dynamic) |
| **Crop Database / Class Labels** | ❌ Not present |

**Conclusion:** There is **no crop recommendation logic (hardcoded, rule-based, or ML-based) currently in the project.** The existing mock data is exclusively tailored to TrustVault security logs.

---

## 5. Python Backend and Machine Learning Model Status

| Component | Searched Formats | Result | Location / Note |
| :--- | :--- | :--- | :--- |
| **Python Backend** | `*.py`, `FastAPI`, `Flask`, `Django` | ❌ **NOT FOUND** | No Python source files in workspace |
| **Trained ML Models** | `*.pkl`, `*.joblib`, `*.h5`, `*.pt`, `*.onnx` | ❌ **NOT FOUND** | No serialized models found |
| **Agricultural Datasets** | `*.csv`, `Crop_recommendation.csv` | ❌ **NOT FOUND** | No agricultural datasets found |
| **Jupyter Notebooks** | `*.ipynb` | ❌ **NOT FOUND** | No training notebooks found |
| **System Python Runtime** | Host executable check | ✅ **FOUND** | `C:\Users\dasar\AppData\Local\Programs\Python\Python313\python.exe` (v3.13.15) |

---

## 6. Proposed Full-Stack Architecture for HARVESTIQ

To satisfy the requirements using **HTML, CSS, JavaScript, Python, and FastAPI**, while preserving the existing codebase and maintaining high-end visual design, the following full-stack structure is proposed:

```
HARVESTIQ/
├── backend/                             # Python FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI application entry point, CORS, exception handlers
│   │   ├── config.py                    # App settings and environment variables
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── crop_schema.py           # Pydantic models (CropInput, CropPredictionResponse)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── ml_service.py            # Model loader, feature preprocessor, inference pipeline
│   │   │   └── advisory_service.py      # Crop care, sowing seasons, water needs, fertilizer tips
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── health.py                # System health, model readiness endpoint
│   │       ├── recommend.py             # POST /api/v1/recommend, GET /api/v1/crops
│   │       └── analytics.py             # Feature importance and parameter boundary endpoints
│   ├── requirements.txt                 # fastapi, uvicorn, scikit-learn, pandas, numpy, joblib
│   └── run.py                           # Development server launch script (uvicorn)
│
├── ml/                                  # Machine Learning Engineering & Training Pipeline
│   ├── data/
│   │   ├── raw/
│   │   │   └── Crop_recommendation.csv  # Standard 2,200-sample benchmark dataset (22 crop classes)
│   │   └── processed/
│   ├── models/
│   │   ├── crop_recommendation_model.joblib # Trained Random Forest / LightGBM pipeline
│   │   ├── label_encoder.joblib         # Categorical label encoder / target classes
│   │   ├── scaler.joblib                # Feature scaler (StandardScaler or MinMaxScaler)
│   │   └── model_metadata.json          # Metrics: accuracy, precision, training timestamp
│   └── train.py                         # Standalone reproducible training script with evaluation
│
├── frontend/                            # HarvestIQ Web Presentation Layer
│   ├── index.html                       # Semantic HTML5 (Hero, Input Console, Result Cards, Advisory)
│   ├── css/
│   │   ├── variables.css                # Design tokens (retaining lush emerald & dark-slate aesthetic)
│   │   ├── base.css                     # Reset, typography, utility classes
│   │   ├── components.css               # Cards, sliders, inputs, badges, dials, modals
│   │   └── layout.css                   # Grid, responsive layout, navigation, sidebar
│   ├── js/
│   │   ├── api.js                       # Asynchronous fetch client communicating with FastAPI
│   │   ├── state.js                     # Client state management (recent predictions, user inputs)
│   │   ├── validation.js                # Input boundary checks (N: 0-140, P: 5-145, K: 5-205, pH: 0-14, etc.)
│   │   ├── charts.js                    # Visual radar/bar charts for soil nutrient balance (Chart.js)
│   │   └── app.js                       # DOM manipulation, event listeners, UI rendering
│   └── assets/                          # Crop iconography, agricultural badges, logos
│
├── tests/                               # Test Suites
│   ├── test_api.py                      # FastAPI endpoint unit & integration tests (pytest + TestClient)
│   └── test_model.py                    # Inference validity and shape consistency tests
│
├── .gitignore
├── README.md                            # Complete setup and execution guide
└── start.bat                            # Windows one-click start script (Backend + Web Server)
```

---

## 7. Communication Architecture: Frontend, Backend, Dataset, and ML Model

```mermaid
flowchart TD
    subgraph Client ["Client Browser (Frontend)"]
        UI["HarvestIQ Web Interface\n(HTML5 + Modern CSS)"]
        VAL["validation.js\n(Range & Type Check)"]
        JS["api.js (Async Fetch)"]
        RENDER["Dynamic DOM & Charts\n(Crop Card + Nutrient Radar)"]
    end

    subgraph Backend ["Python FastAPI Application"]
        ROUTER["Router: /api/v1/recommend"]
        PYDANTIC["Pydantic Validation\n(CropInputSchema)"]
        ML_SERV["ml_service.py\n(Inference Engine)"]
        ADV_SERV["advisory_service.py\n(Agronomic Intelligence)"]
    end

    subgraph ML_Layer ["Machine Learning Model Artifacts"]
        PIPELINE["Model Pipeline (joblib)\nRandomForest / LightGBM"]
        SCALER["StandardScaler\n(Feature Scaling)"]
        ENCODER["LabelEncoder\n(22 Crop Classes)"]
    end

    subgraph Data_Layer ["Offline Training Pipeline"]
        DATASET["Crop_recommendation.csv\n(2200 samples, 7 features)"]
        TRAIN["train.py\n(Stratified K-Fold CV)"]
    end

    %% Training relationship
    DATASET --> TRAIN
    TRAIN -->|Exports Serialized Models| ML_Layer

    %% Runtime Request Flow
    UI -->|Farmer inputs N, P, K, Temp, Humidity, pH, Rainfall| VAL
    VAL -->|Sanitized Form Data| JS
    JS -->|HTTP POST JSON Payload| ROUTER
    ROUTER --> PYDANTIC
    PYDANTIC --> ML_SERV

    %% Inference Flow
    ML_SERV -->|Features [N,P,K,T,H,pH,R]| SCALER
    SCALER -->|Normalized Features| PIPELINE
    PIPELINE -->|Class Index & Probabilities| ENCODER
    ENCODER -->|Predicted Crop Name + Top 3 Probabilities| ML_SERV

    %% Advisory & Response Flow
    ML_SERV --> ADV_SERV
    ADV_SERV -->|Enriched Agronomic Context| ROUTER
    ROUTER -->|HTTP 200 JSON Response| JS
    JS --> RENDER
```

### Detailed Communication Sequence:
1. **User Interaction & Client-Side Validation:**
   - The user inputs the 7 critical agro-ecological parameters:
     - **Nitrogen ($N$):** Ratio in soil ($0 - 140\text{ kg/ha}$)
     - **Phosphorus ($P$):** Ratio in soil ($5 - 145\text{ kg/ha}$)
     - **Potassium ($K$):** Ratio in soil ($5 - 205\text{ kg/ha}$)
     - **Temperature:** Celsius ($8^\circ\text{C} - 50^\circ\text{C}$)
     - **Humidity:** Relative percentage ($10\% - 100\%$)
     - **Soil pH:** Acidity/Alkalinity level ($3.5 - 9.5$)
     - **Rainfall:** Precipitation ($20\text{ mm} - 300\text{ mm}$)
   - `validation.js` checks bounds and highlights any physiological anomalies before dispatching the network request.

2. **Network Transmission (REST API):**
   - `api.js` issues a `POST` request to `http://localhost:8000/api/v1/recommend` with payload:
     ```json
     {
       "nitrogen": 90.0,
       "phosphorus": 42.0,
       "potassium": 43.0,
       "temperature": 20.87,
       "humidity": 82.0,
       "ph": 6.5,
       "rainfall": 202.93
     }
     ```

3. **Backend Processing & Pydantic Validation:**
   - FastAPI intercepts the request.
   - Pydantic validates data types and ranges. If any value falls outside physiological limits, FastAPI automatically returns a structured `422 Unprocessable Entity` response with specific field error details.

4. **Machine Learning Inference Pipeline:**
   - `ml_service.py` receives the validated parameters and constructs a 2D NumPy array: `[[N, P, K, temperature, humidity, ph, rainfall]]`.
   - The array passes through the pre-fitted feature scaler (if applicable).
   - The trained ensemble model (Random Forest / LightGBM) calls `predict()` to identify the most suitable crop class and `predict_proba()` to compute probability distributions across all 22 crop classes.
   - The top 3 alternative crops with their respective confidence percentages are extracted.

5. **Agronomic Advisory Enrichment:**
   - `advisory_service.py` matches the predicted crop against agricultural knowledge bases to append:
     - Ideal sowing season (e.g., Kharif, Rabi, Zaid).
     - Water requirement (Low, Moderate, High).
     - Soil health evaluation (e.g., Nitrogen surplus, Potassium deficiency warning).
     - Fertilizer adjustment recommendations.

6. **Structured Response Delivery & Dynamic Rendering:**
   - FastAPI responds with HTTP 200:
     ```json
     {
       "success": true,
       "prediction": {
         "recommended_crop": "rice",
         "confidence_score": 98.4,
         "alternative_crops": [
           { "crop": "jute", "confidence": 1.2 },
           { "crop": "coffee", "confidence": 0.4 }
         ]
       },
       "agronomic_advisory": {
         "sowing_season": "Kharif (June - October)",
         "water_requirement": "High (Standing water)",
         "soil_summary": "Optimal N-P-K balance for cereal grain cultivation."
       }
     }
     ```
   - `app.js` renders the recommended crop card with badge iconography, confidence meter, and soil radar charts without requiring a full page refresh.

---

## 8. Preserving Existing Branding & Assets

- The existing CSS system (`App.css`) contains over 4,600 lines of production-grade UI components, dark-slate surfaces (`#070c16`, `#0c1424`), and emerald accents (`#10b981`).
- This design palette is an exceptional aesthetic fit for an agricultural intelligence platform, where emerald greens signify crop vitality and dark-slate gives an executive, modern SaaS dashboard feel.
- In subsequent phases, we can seamlessly adapt these UI components and card containers to host the HarvestIQ inputs, prediction dials, and agricultural charts, avoiding redundant design work while elevating the visual quality.

---

## 9. Next Steps Development Plan

1. **Dataset Acquisition & Exploration:**
   - Acquire the benchmark Crop Recommendation dataset (2,200 records across 22 crops: Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee).
2. **Machine Learning Pipeline Development:**
   - Build a reproducible `train.py` script.
   - Train and benchmark multiple classifiers (Random Forest, XGBoost/LightGBM, SVM).
   - Export the highest-performing model (`.joblib`) with cross-validated accuracy (>99%).
3. **FastAPI Backend Construction:**
   - Set up the FastAPI server with Pydantic request/response validation.
   - Implement `/api/v1/recommend`, `/api/v1/crops`, and `/api/v1/health` endpoints.
4. **HarvestIQ Frontend Implementation:**
   - Create the clean HTML5, modern CSS (leveraging the existing design system), and JavaScript interface with interactive sliders, real-time validation, and Chart.js integration.
5. **End-to-End Integration & Verification:**
   - Verify model inference latency, error handling, and cross-browser responsiveness.
