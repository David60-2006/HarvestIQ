# HARVESTIQ – Machine Learning Model Training & Evaluation Report

**Project:** HARVESTIQ – An Intelligent Crop Recommendation System Using Machine Learning  
**Date:** November 2026  
**Author:** Machine Learning Engineer & Data Scientist  
**Model Artifact Saved:** `backend/models/harvestiq_model.joblib`  
**Winning Algorithm:** **RandomForestClassifier**  

---

## 1. Dataset & Experimental Setup

* **Dataset Path:** `backend/data/Crop_recommendation.csv`
* **Total Instances:** 2200 samples
* **Features (7 continuous inputs):**
  - Soil Nitrogen (N, kg/ha)
  - Soil Phosphorus (P, kg/ha)
  - Soil Potassium (K, kg/ha)
  - Ambient Temperature (°C)
  - Relative Humidity (%)
  - Soil pH (3.0 - 10.0)
  - Seasonal Rainfall (mm)
* **Target Classes:** 22 distinct crops (perfectly balanced at 100 samples/class)
* **Data Splitting:**
  - 80% Stratified Training Set (1,760 samples)
  - 20% Stratified Holdout Test Set (440 samples)
  - Reproducible `random_state = 42`
* **Zero Data Leakage Protocol:** Preprocessing (`StandardScaler`) is encapsulated inside `sklearn.pipeline.Pipeline`, ensuring scaling statistics are fitted strictly on the training partitions and only transformed onto validation and test folds.

---

## 2. Algorithms Evaluated

Five diverse algorithms spanning non-parametric, linear, instance-based, and ensemble techniques were benchmarked:

1. **DummyClassifier (Baseline):** Predicts class prior distributions (`strategy="prior"`). Serves as a baseline benchmark (4.55% random chance).
2. **Logistic Regression:** Regularized linear model with standard feature scaling (`max_iter=1000`).
3. **K-Nearest Neighbors (KNN):** Distance-weighted instance classifier (`k=5`) with scaled features.
4. **Random Forest Classifier:** Bagged ensemble of 100 decorrelated decision trees with parallelized sub-sampling.
5. **Gradient Boosting Classifier:** Boosted ensemble sequentially minimizing multi-class cross-entropy loss (100 estimators, learning rate 0.1, depth 3).

---

## 3. Empirical Model Comparison

Cross-validation was conducted using **5-Fold Stratified K-Fold** on the 1,760 training samples. Final generalization was validated on the 440 holdout test samples.

| Algorithm | CV_Accuracy_Mean | CV_Accuracy_Std | CV_Macro_F1_Mean | CV_Macro_F1_Std | Test_Accuracy | Test_Precision_Macro | Test_Recall_Macro | Test_F1_Macro | Test_Top3_Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DummyClassifier | 0.0455 | 0.0000 | 0.0040 | 0.0000 | 0.0455 | 0.0021 | 0.0455 | 0.0040 | 0.1364 |
| LogisticRegression | 0.9682 | 0.0066 | 0.9678 | 0.0068 | 0.9727 | 0.9740 | 0.9727 | 0.9725 | 1.0000 |
| KNeighborsClassifier | 0.9653 | 0.0121 | 0.9650 | 0.0124 | 0.9795 | 0.9804 | 0.9795 | 0.9793 | 0.9977 |
| RandomForestClassifier | 0.9926 | 0.0058 | 0.9926 | 0.0059 | 0.9955 | 0.9957 | 0.9955 | 0.9955 | 1.0000 |
| GradientBoostingClassifier | 0.9869 | 0.0034 | 0.9869 | 0.0036 | 0.9886 | 0.9897 | 0.9886 | 0.9887 | 0.9977 |

### Metric Definitions:
* **CV Accuracy / Macro F1 (Mean ± Std):** 5-fold stratified cross-validation on training data, reflecting out-of-fold generalization.
* **Test Accuracy:** Overall percentage of correct predictions on unseen test data.
* **Test F1 Macro:** Unweighted average of F1 scores across all 22 classes (equal weighting for all crop types).
* **Test Top-3 Accuracy:** Percentage of test instances where the ground-truth crop was ranked in the top 3 predicted candidates.

---

## 4. Best Model Selection & Justification

### Selected Model: **RandomForestClassifier**

* **Out-of-Fold Cross-Validation Accuracy:** **99.26% ± 0.58%**
* **Holdout Test Accuracy:** **99.55%**
* **Holdout Macro F1 Score:** **99.55%**
* **Holdout Top-3 Accuracy:** **100.00%**

### Architectural Justification:
1. **Superior Non-Linear Decision Boundaries:** Agro-climatic relationships exhibit strong non-linear thresholds (e.g., rice requires high rainfall >200mm AND moderate temperatures; cotton requires high potassium). Tree-based ensemble partitioning captures these multi-dimensional boundary conditions far better than linear approximations.
2. **Cross-Validation Stability:** The winning model demonstrated negligible fold variance (± 0.58%), proving the absence of fold overfitting.
3. **Robustness to Extreme Outliers:** Random subsampling and tree ensembling offer natural resistance to soil nutrient variance without suffering from distance distortion (which limits KNN in higher dimensions).
4. **Calibrated Probabilities:** The ensemble generates meaningful class probability distributions, enabling the FastAPI backend to reliably return top-3 alternative crops and confidence metrics to farmers.

---

## 5. Per-Crop Classification Performance (Winning Model)

Evaluation on the 440 holdout test samples (20 samples per crop class):

| Crop | Precision | Recall | F1_Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| apple | 1.0000 | 1.0000 | 1.0000 | 20 |
| banana | 1.0000 | 1.0000 | 1.0000 | 20 |
| blackgram | 1.0000 | 0.9500 | 0.9744 | 20 |
| chickpea | 1.0000 | 1.0000 | 1.0000 | 20 |
| coconut | 1.0000 | 1.0000 | 1.0000 | 20 |
| coffee | 1.0000 | 1.0000 | 1.0000 | 20 |
| cotton | 1.0000 | 1.0000 | 1.0000 | 20 |
| grapes | 1.0000 | 1.0000 | 1.0000 | 20 |
| jute | 0.9524 | 1.0000 | 0.9756 | 20 |
| kidneybeans | 1.0000 | 1.0000 | 1.0000 | 20 |
| lentil | 1.0000 | 1.0000 | 1.0000 | 20 |
| maize | 0.9524 | 1.0000 | 0.9756 | 20 |
| mango | 1.0000 | 1.0000 | 1.0000 | 20 |
| mothbeans | 1.0000 | 1.0000 | 1.0000 | 20 |
| mungbean | 1.0000 | 1.0000 | 1.0000 | 20 |
| muskmelon | 1.0000 | 1.0000 | 1.0000 | 20 |
| orange | 1.0000 | 1.0000 | 1.0000 | 20 |
| papaya | 1.0000 | 1.0000 | 1.0000 | 20 |
| pigeonpeas | 1.0000 | 1.0000 | 1.0000 | 20 |
| pomegranate | 1.0000 | 1.0000 | 1.0000 | 20 |
| rice | 1.0000 | 0.9500 | 0.9744 | 20 |
| watermelon | 1.0000 | 1.0000 | 1.0000 | 20 |

* **Macro Average Precision:** 99.57%
* **Macro Average Recall:** 99.55%
* **Macro Average F1-Score:** 99.55%

---

## 6. Generated Artifacts & Reports

The following files were created in the repository:

1. **`backend/models/harvestiq_model.joblib`**: Serialized production pipeline containing scaler, fitted classifier, and class label metadata.
2. **`backend/models/crop_recommendation_model.joblib`**: Mirrored model artifact ensuring backward compatibility with FastAPI `PredictionService`.
3. **`backend/reports/model_comparison.csv`**: Comprehensive comparison matrix across all 5 benchmarked algorithms.
4. **`backend/reports/classification_report.csv`**: Granular precision, recall, and F1 scores for each of the 22 crops.
5. **`backend/reports/confusion_matrix.png`**: High-resolution annotated 22×22 confusion matrix visualization.
6. **`backend/reports/evaluation_report.json`**: Machine-readable JSON report with parameters, cross-validation metrics, and deployment readiness metadata.

---

## 7. FastAPI Backend Compatibility & Next Steps

* The model pipeline is serialized with both the feature transformer and classifier in a single object (`Pipeline`).
* Raw inputs `[N, P, K, temperature, humidity, ph, rainfall]` can be passed directly to `pipeline.predict()` and `pipeline.predict_proba()` without external manual scaling steps.
* The model artifact is verified and ready for live serving via `POST /recommend`.
