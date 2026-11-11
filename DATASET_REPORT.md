# HARVESTIQ – Dataset Analysis & Verification Report

**Project:** HARVESTIQ – An Intelligent Crop Recommendation System Using Machine Learning  
**Dataset Name:** Crop Recommendation Dataset  
**Original Author:** Atharva Ingle (Benchmark Agricultural Dataset)  
**License:** CC0 1.0 Universal (Public Domain)  
**File Verified:** `C:\Users\dasar\trustvault\backend\data\raw\Crop_recommendation.csv`  
**Generated Date:** November 2026  

---

## 1. Provenance & Dataset Verification

| Property | Verified Value |
| :--- | :--- |
| **Origin Source** | Kaggle Crop Recommendation Dataset / Verified Agricultural Repository |
| **License Type** | CC0: Public Domain (Open for academic, research & commercial use) |
| **Raw File Size** | 144.37 KB (147833 bytes) |
| **SHA-256 Checksum** | `808e1d84a544647f1ab2fdae04f434b1842692d6be09eee18a5c2d43eb78d77c` |
| **Total Record Count** | **2200 rows** (Zero artificial or fabricated records) |
| **Feature Dimension** | **8 columns** (7 agro-climatic predictors + 1 categorical target label) |
| **Data Integrity** | **100% complete** (0 null values, 0 duplicates) |

---

## 2. Schema & Data Types

The dataset contains the exact 7 physiological indicators required for precision crop recommendation:

| Column | Standardized Name | Data Type | Physical Meaning | Valid Range |
| :--- | :--- | :--- | :--- | :--- |
| `N` | `n` | `float64` / `int64` | Ratio of Nitrogen in soil | 0 – 140 kg/ha |
| `P` | `p` | `float64` / `int64` | Ratio of Phosphorus in soil | 5 – 145 kg/ha |
| `K` | `k` | `float64` / `int64` | Ratio of Potassium in soil | 5 – 205 kg/ha |
| `temperature` | `temperature` | `float64` | Ambient temperature | 8.83 – 43.68 °C |
| `humidity` | `humidity` | `float64` | Relative humidity percentage | 14.26 – 99.98 % |
| `ph` | `ph` | `float64` | Soil pH level (acidity/alkalinity) | 3.50 – 9.94 |
| `rainfall` | `rainfall` | `float64` | Seasonal precipitation | 20.21 – 298.56 mm |
| `label` | `label` | `object` (string) | Target crop classification | 22 unique classes |

---

## 3. Data Hygiene & Quality Checks

1. **Missing Value Audit:**
   - Evaluated across all columns: zero missing or `NaN` values detected.
2. **Duplicate Record Audit:**
   - Evaluated across all 8 dimensions: zero duplicate rows detected.
3. **Boundary Consistency Check:**
   - All feature observations fall strictly within the physiological boundaries configured in the HarvestIQ FastAPI validation schema:
     - $N \in [0, 200]$, $P \in [0, 200]$, $K \in [0, 300]$
     - $\text{Temperature} \in [0, 50]$, $\text{Humidity} \in [0, 100]$
     - $\text{pH} \in [3, 10]$, $\text{Rainfall} \in [0, 400]$
4. **Data Separation:**
   - Raw source preserved at: `backend/data/raw/Crop_recommendation.csv`
   - Cleaned & normalized version exported to: `backend/data/processed/Crop_recommendation_cleaned.csv`
   - Zero modifications or artificial records were injected into the raw dataset.

---

## 4. Descriptive Statistical Profile

| Feature | count | mean | std | min | 25% | 50% | 75% | max | median | iqr | skew |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| n | 2200.00 | 50.55 | 36.92 | 0.00 | 21.00 | 37.00 | 84.25 | 140.00 | 37.00 | 63.25 | 0.51 |
| p | 2200.00 | 53.36 | 32.99 | 5.00 | 28.00 | 51.00 | 68.00 | 145.00 | 51.00 | 40.00 | 1.01 |
| k | 2200.00 | 48.15 | 50.65 | 5.00 | 20.00 | 32.00 | 49.00 | 205.00 | 32.00 | 29.00 | 2.38 |
| temperature | 2200.00 | 25.62 | 5.06 | 8.83 | 22.77 | 25.60 | 28.56 | 43.68 | 25.60 | 5.79 | 0.18 |
| humidity | 2200.00 | 71.48 | 22.26 | 14.26 | 60.26 | 80.47 | 89.95 | 99.98 | 80.47 | 29.69 | -1.09 |
| ph | 2200.00 | 6.47 | 0.77 | 3.50 | 5.97 | 6.43 | 6.92 | 9.94 | 6.43 | 0.95 | 0.28 |
| rainfall | 2200.00 | 103.46 | 54.96 | 20.21 | 64.55 | 94.87 | 124.27 | 298.56 | 94.87 | 59.72 | 0.97 |

### Statistical Observations:
* **Soil Nutrients ($N, P, K$):** High variability across crop types. Potassium ($K$) and Phosphorus ($P$) show positive skewness driven by specific potassium-intensive crops (such as apples, grapes, and bananas).
* **Soil pH:** Centered at mean $\mu = 6.47$ with median $6.43$, representing near-neutral agricultural arable land, with range spanning acidic ($3.50$) to alkaline ($9.94$).
* **Rainfall:** Ranges from $20.21\text{ mm}$ (arid crops such as muskmelon and chickpea) to $298.56\text{ mm}$ (heavy monsoon crops like rice and jute).
* **Relative Humidity:** Spans from dry arid conditions ($14.26\%$) to near-saturation ($99.98\%$).

---

## 5. Crop Class Distribution Analysis

* **Total Classes:** 22 unique crops
* **Class Balance:** **Perfect balance** — exactly 100 observations per crop class.
* **Balanced Classification Ratio:** $4.55\%$ per class across the entire 2,200 sample space.
* **Crops Included:**  
  `apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`

---

## 6. Generated Exploratory Data Analysis (EDA) Visualizations

The following high-resolution charts were generated and saved in `backend/data/eda_charts/`:

- **Crop Distribution:** `backend/data/eda_charts/crop_distribution.png`
- **Feature Distributions:** `backend/data/eda_charts/feature_distributions.png`
- **Correlation Heatmap:** `backend/data/eda_charts/correlation_heatmap.png`
- **Npk By Crop:** `backend/data/eda_charts/npk_by_crop.png`
- **Climate Vs Rainfall:** `backend/data/eda_charts/climate_vs_rainfall.png`

### Key EDA Insights:
1. **Class Uniformity:** The horizontal bar chart confirms uniform sampling across all 22 crop classes, preventing majority-class bias during model training.
2. **Feature Collinearity:** Pearson correlation analysis shows a notable positive correlation between Phosphorus ($P$) and Potassium ($K$) ($r \approx 0.74$), which aligns agronomically with fruit crops requiring balanced PK fertilization.
3. **Climatic Clustering:** Scatter analysis of rainfall vs. temperature reveals clear natural clustering (e.g., Rice and Jute occupy the high-rainfall corridor >200mm, whereas Chickpea and Kidneybeans cluster in low-humidity, moderate-temperature zones).

---

## 7. Preprocessing & Data Leakage Prevention

* **Separation of Concerns:** Feature transformation pipelines (`StandardScaler`, `LabelEncoder`) must only be fitted on the **training fold** ($X_{\text{train}}$) during cross-validation, and never on the full dataset, preventing lookahead data leakage.
* **Stratified Splitting:** Given the multi-class setup (22 classes), stratified 5-fold cross-validation or stratified train-test splits ($80/20$) will ensure each crop has proportional representation in validation sets.

---

## 8. Agronomic Scope & Regional Notice

> [!IMPORTANT]
> **Agronomic Limitation & Regional Scope:**  
> While this benchmark dataset provides an empirical foundation for multi-class agro-climatic machine learning modeling, it is **not claimed to be scientifically validated for every micro-climatic agro-ecological zone in India or globally**. Factors such as local soil microbiology, depth of topsoil, sunlight hours (photoperiod), pest pressures, irrigation methods, and seasonal monsoon shifts vary across agro-climatic zones and should complement machine learning predictions with localized agronomic advisory.
