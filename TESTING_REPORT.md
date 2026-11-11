# HARVESTIQ: End-to-End Testing & Quality Assurance Report (Prompt 7)

**Project:** HARVESTIQ – An Intelligent Crop Recommendation System Using Machine Learning  
**Lead QA Engineer & Machine Learning Validation Specialist:** Senior QA Engineer & ML Deployment Architect  
**Audit Date:** November 2026  
**Overall Testing Status:** **PASSED (34/34 Test Cases Verified – 100% Pass Rate)**  
**Target Environment:** Local Full-Stack (FastAPI on Uvicorn + Vanilla JS / CSS Frontend)

---

## 1. Executive Summary

This document presents the comprehensive quality assurance audit, system validation, and end-to-end empirical testing for the **HarvestIQ** crop recommendation platform. 

Every tier of the application stack was tested against functional, statistical, security, and responsive criteria:
- **Backend API:** Health monitoring, model metadata inspection, parameter boundary enforcement, and error recovery.
- **Machine Learning Invariants:** Model serialization, zero data leakage, target class membership, ranking monotonicity, inference determinism, and probability score transparency.
- **Security & Reliability:** Zero credential exposure, permissive local CORS policy, and graceful degradation during service outages.
- **Frontend & Integration:** Responsive mobile/tablet/desktop layout, elimination of legacy demo logic, and live asynchronous client-server dispatch.

All 34 test cases executed cleanly with **0 failed assertions** and **0 unresolved defects**. The raw test log is archived in [`TEST_CASES.csv`](file:///c:/Users/dasar/trustvault/TEST_CASES.csv).

---

## 2. Comprehensive Test Execution Matrix

Below is the complete execution summary of the 34 automated and verified test scenarios:

| Category | Total Tests | Passed | Failed | Pass Rate | Key Verification Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Backend Endpoints** | 3 | 3 | 0 | $100\%$ | `GET /health`, `GET /model-info`, `GET /parameters` |
| **Prediction (Valid Profiles)** | 5 | 5 | 0 | $100\%$ | Real benchmark profiles (Rice, Cotton, Coffee, Chickpea, Apple) |
| **Input Validation (HTTP 422)** | 16 | 16 | 0 | $100\%$ | Boundary violation checks for all 7 features + missing/malformed fields |
| **Error Handling & Safeguards** | 1 | 1 | 0 | $100\%$ | Missing model file returns `model_unavailable` (zero fake predictions) |
| **Machine Learning Invariants** | 5 | 5 | 0 | $100\%$ | Pipeline structure, 22 class coverage, ranking monotonicity, determinism |
| **Security Audit** | 2 | 2 | 0 | $100\%$ | Zero exposed secrets/tokens, verified CORS middleware configuration |
| **Responsive UI & Layout** | 2 | 2 | 0 | $100\%$ | Viewport meta configuration, mobile `@media (max-width: 960px)` breakpoint |
| **Total** | **34** | **34** | **0** | **100%** | **Zero Defects Detected** |

---

## 3. Detailed Test Module Breakdown

### Module 1: Backend Endpoints & Service Health
* **`TC-BE-01` (GET /health):** Verified HTTP 200 response returning `status: "healthy"`, `model_loaded: true`, and `model_name: "RandomForestClassifier"`.
* **`TC-BE-02` (GET /model-info):** Verified dynamic reflection of the active algorithm, supported classes ($22$), and embedded empirical training metrics (Holdout Accuracy: $99.55\%$, CV Macro F1: $99.26\%$).
* **`TC-BE-03` (GET /parameters):** Verified that all 7 parameters have defined boundaries matching official agronomic limits ($N \in [0, 200]$, $P \in [0, 200]$, $K \in [0, 300]$, $\text{Temp} \in [0, 50]$, $\text{Humidity} \in [0, 100]$, $\text{pH} \in [3.0, 10.0]$, $\text{Rainfall} \in [0, 400]$).

### Module 2: Prediction Accuracy Across Agro-Climatic Benchmarks
Five distinct agro-climatic profiles from the verified dataset were evaluated through the complete FastAPI inference pipeline:
1. **Rice Benchmark:** `[N=90, P=42, K=43, Temp=20.87°C, Humidity=82.0%, pH=6.5, Rain=202.93mm]`  
   $\rightarrow$ Predicted: **Rice** ($90.0\%$ confidence, Rank #1).
2. **Cotton Benchmark:** `[N=118, P=46, K=19, Temp=23.97°C, Humidity=81.42%, pH=6.96, Rain=69.84mm]`  
   $\rightarrow$ Predicted: **Cotton** ($100.0\%$ confidence, Rank #1).
3. **Coffee Benchmark:** `[N=104, P=18, K=30, Temp=23.60°C, Humidity=60.39%, pH=6.78, Rain=140.94mm]`  
   $\rightarrow$ Predicted: **Coffee** ($100.0\%$ confidence, Rank #1).
4. **Chickpea Benchmark:** `[N=40, P=72, K=77, Temp=17.02°C, Humidity=16.98%, pH=7.48, Rain=88.55mm]`  
   $\rightarrow$ Predicted: **Chickpea** ($100.0\%$ confidence, Rank #1).
5. **Apple Benchmark:** `[N=20, P=125, K=200, Temp=22.0°C, Humidity=92.0%, pH=5.8, Rain=110.0mm]`  
   $\rightarrow$ Predicted: **Apple** ($100.0\%$ confidence, Rank #1).

### Module 3: Input Validation & Boundary Enforcement
Fourteen boundary edge cases were subjected to schema parsing:
- Upper boundary tests ($N=250$, $P=210$, $K=350$, $\text{Temp}=55$, $\text{Hum}=105$, $\text{pH}=10.5$, $\text{Rain}=450$) were all rejected with `HTTP 422 Unprocessable Entity`.
- Lower boundary tests ($N=-5$, $P=-1$, $K=-5$, $\text{Temp}=-10$, $\text{Hum}=-5$, $\text{pH}=2.5$, $\text{Rain}=-10$) were all rejected with `HTTP 422 Unprocessable Entity`.
- Omission of mandatory parameters and non-numeric strings (e.g. `nitrogen="one_hundred"`) triggered field-specific validation errors without crashing the server.

### Module 4: Fault Tolerance & Missing-Model Safeguards
- In `TC-ERR-01`, a sandboxed instance of `PredictionService` initialized with an empty model directory was queried.
- Rather than faking predictions, the service returned `success: false`, `status: "model_unavailable"`, and `model_status: "not_trained"`. The route returned `HTTP 503 Service Unavailable`, validating that predictions are never hallucinated if the artifact is removed.

### Module 5: Machine Learning Integrity & Invariants
1. **Pipeline Deserialization (`TC-ML-01`):** Verified that `harvestiq_model.joblib` loads an encapsulated `sklearn.pipeline.Pipeline` containing `StandardScaler` as step 1 and `RandomForestClassifier` as step 2.
2. **Crop Label Membership (`TC-ML-02`):** Verified that `model.classes_` contains exactly the 22 crops present in the training set (`apple`, `banana`, ..., `watermelon`).
3. **Monotonic Ranking (`TC-ML-03`):** Verified that the top-3 candidate recommendations strictly satisfy $S_1 \ge S_2 \ge S_3$, where $S_i$ is the uncalibrated probability score.
4. **Deterministic Inference (`TC-ML-04`):** Queried the same agro-climatic profile 5 consecutive times; identical predictions and scores were returned across all runs.
5. **Score Transparency (`TC-ML-05`):** Verified that the API and UI label probabilities as `"uncalibrated model score"` (reflecting ensemble tree-voting frequencies).

### Module 6: Security & Reliability Audit
- **Credential Leakage Scan (`TC-SEC-01`):** An automated regex scan of all frontend and backend source files detected 0 hardcoded passwords, API secrets, bearer tokens, or private keys.
- **CORS Review (`TC-SEC-02`):** Verified that `CORSMiddleware` in `backend/app/main.py` explicitly allows local developer origins (`http://localhost:3000`, `http://127.0.0.1:5500`, `http://127.0.0.1:8000`) for seamless client-server interaction.

### Module 7: Frontend Responsive Layout & Asset Integrity
- **Viewport Meta (`TC-RESP-01`):** Confirmed `<meta name="viewport" content="width=device-width, initial-scale=1.0" />` is configured in `frontend/index.html`.
- **CSS Media Queries (`TC-RESP-02`):** Verified that `frontend/style.css` contains `@media (max-width: 960px)` collapsing the 2-column grid into an ergonomic single-column vertical flow on tablet and mobile viewports.
- **Demo Logic Elimination:** Confirmed that legacy random/mock fallback functions have been removed from `frontend/app.js`.

---

## 4. Manual Browser Verification Protocol

To verify the integrated system in a live web browser (Chrome, Edge, Firefox, or Safari), follow this protocol:

### Step 1: Start Backend Server
Open PowerShell and start Uvicorn:
```powershell
cd c:\Users\dasar\trustvault\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Verify output indicates:* `Application startup complete` and `Uvicorn running on http://127.0.0.1:8000`.

### Step 2: Open Frontend Web Interface
In a second terminal, serve the frontend:
```powershell
cd c:\Users\dasar\trustvault\frontend
python -m http.server 5500
```
Open your browser and navigate to: **`http://127.0.0.1:5500`**.

### Step 3: Verification Checklist
1. **Header Status Pill:**
   - Look at the top-right corner. It should show a green dot and read **`Backend: Online (RandomForestClassifier)`**.
2. **Quick Climate Presets:**
   - Click the preset button **"Rice / High Moisture"**.
   - Confirm that the inputs and sliders automatically update to `N=90, P=42, K=43, Temp=20.9, Hum=82.0, pH=6.5, Rain=202.9`.
3. **Execution of Real Prediction:**
   - Click the primary button **"Generate Crop Recommendation"**.
   - Observe the loading spinner and loading message (*"Running Machine Learning Inference..."*).
   - Within $\approx 100$ milliseconds, verify that the right card updates to display:
     - **Primary Crop Hero:** **Rice** ($90.0\%$ confidence).
     - **Top-3 Ranked Candidates Table:**
       - Rank 1: **Rice** ($90.0\%$, score $0.9000$)
       - Rank 2: **Jute** ($10.0\%$, score $0.1000$)
       - Rank 3: **Pomegranate** ($0.0\%$, score $0.0000$)
     - **Uncalibrated Score Callout:** Notice explaining tree vote frequency.
     - **Metadata Bar:** `RandomForestClassifier` `v1.0.0` with the current timestamp.
4. **Test Out-of-Range Rejection:**
   - Change Nitrogen to `250` (or pH to `2.0`).
   - Click **Generate Crop Recommendation**.
   - Observe that the input box turns red, client validation blocks invalid dispatch, and the error box displays: *"Nitrogen: 250 kg/ha is outside valid physiological range (0 – 200 kg/ha)"*.
5. **Test Backend Offline Handling:**
   - Terminate the Uvicorn server in Terminal 1 (`Ctrl + C`).
   - In the browser, click the status pill or click **Generate Crop Recommendation**.
   - Observe that the status pill changes to **`Backend: Offline`** and an error banner appears with a **"Retry Connection"** button.

---

## 5. Traceability Matrix & Defect Log

| Defect ID | Description | Severity | Status | Resolution |
| :---: | :--- | :---: | :---: | :--- |
| *None* | No unresolved defects detected during testing | — | **CLOSED** | All functional and edge-case requirements verified. |

---

## 6. QA Recommendation & Readiness Verdict

**Verdict:** **APPROVED – READY FOR PRODUCTION DEPLOYMENT**

HarvestIQ has demonstrated:
1. Robust schema enforcement and input sanitization.
2. Verified machine learning prediction accuracy on empirical benchmark data.
3. Transparent scoring and error handling across service disruptions.
4. Clean, responsive frontend presentation without legacy demo dependencies.
