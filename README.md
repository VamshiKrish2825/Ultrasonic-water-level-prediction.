# 🔊 Ultrasonic Water Level Prediction — Machine Learning Pipeline

> **Final Year Project** — A complete, production-grade ML pipeline for predicting water volume (litres) from ultrasonic sensor measurements across multiple frequencies.

---

## 📋 Project Overview

This project develops a machine learning system to predict **water volume (in litres)** from raw ultrasonic sensor measurements. The sensor records wave amplitude, time-of-flight, and reflection features across multiple frequency bands (kHz). The pipeline covers every stage of a professional ML workflow — from data cleaning to ensemble modelling and deployment.

---

## 🏆 Final Results

| Metric | Value |
|--------|-------|
| **Best Model** | Stacking Ensemble (Extra Trees + Gradient Boosting + Random Forest + AdaBoost + Decision Tree) |
| **R² Score** | **0.9951** |
| **Bootstrap 95% CI** | [0.9898 – 0.9972] |
| **MAE** | 0.6042 L |
| **RMSE** | 0.7039 L |
| **MAPE** | 12.65% |
| **Outlier Strategy** | IQR Cleaning (39 rows removed from 144) |
| **Train / Test Split** | 84 / 21 (Frequency-Stratified) |

---

## 📁 Repository Structure

```
├── Ultrasonic_water_level_FINAL_v3.ipynb   # Main notebook — full pipeline
├── Features.csv                             # Raw sensor dataset
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
│
├── water_volume_model_final.pkl             # Trained model (joblib)
├── water_volume_model_metadata.json         # Model metadata & metrics
│
├── results_baseline_models.csv             # All 12 baseline model results
├── results_cv_scores.csv                   # 5-Fold CV scores per model
├── results_tuning.csv                      # Hyperparameter tuning results
├── results_final_comparison.csv            # Baseline + Tuned + Ensemble
├── results_feature_importance.csv          # Permutation feature importance
├── results_feature_selection.csv           # Feature count sweep results
├── results_calibration.csv                 # MAE per volume range
└── results_cleaning.csv                    # Outlier strategy comparison
```

---

## 🔁 Pipeline Summary

```
Raw Data (144 rows × 14 columns)
    │
    ├── EDA: Distributions, Correlations, Pairplot
    │
    ├── Outlier Detection & Removal
    │       ├── IQR Method
    │       ├── Isolation Forest
    │       └── Z-Score  →  Best strategy selected automatically
    │
    ├── Feature Engineering
    │       ├── 66 Ratio Features
    │       ├── 3 Polynomial (Squared) Features
    │       └── 1 Interaction Feature  →  82 total features
    │
    ├── Frequency-Stratified Train/Test Split (80/20)
    │
    ├── Baseline Model Benchmarking (12 models)
    │       ├── Linear: RobustScaler + PowerTransformer → original 13 features
    │       └── Trees:  No scaler → all 82 engineered features
    │
    ├── 5-Fold Cross-Validation
    │
    ├── Hyperparameter Optimisation
    │       ├── RandomizedSearchCV  (wide exploration, 50 iterations)
    │       └── GridSearchCV        (fine-tuning around best region)
    │
    ├── Ensemble Learning
    │       ├── Voting Regressor    (R² = 0.9906)
    │       └── Stacking Regressor  (R² = 0.9951) ← Winner
    │
    ├── In-Depth Analysis
    │       ├── Learning Curve
    │       ├── Bootstrap Confidence Intervals
    │       ├── Permutation Feature Importance
    │       ├── Built-in Feature Importance
    │       ├── SHAP Values (Bar + Beeswarm)
    │       └── Partial Dependence Plots (1D + 2D)
    │
    ├── Diagnostics
    │       ├── Residual Plots (4-panel)
    │       ├── Actual vs Predicted
    │       ├── Error by Frequency
    │       ├── Calibration by Volume Range
    │       ├── Prediction Intervals
    │       └── Statistical Tests (Shapiro-Wilk, D'Agostino, t-test, VIF)
    │
    └── Model Export & Deployment Function
```

---

## 📊 Model Performance Summary

| Model | R² | MAE (L) | RMSE (L) | Features Used |
|-------|----|---------|----------|---------------|
| Stacking Ensemble ⭐ | **0.9951** | **0.6042** | **0.7039** | 82 (trees) |
| Extra Trees (Tuned) | 0.9939 | 0.6622 | 0.7825 | 82 (trees) |
| Gradient Boosting (Tuned) | 0.9935 | — | — | 82 (trees) |
| SVR | 0.9926 | 0.6804 | 0.8627 | 13 (original) |
| Linear Regression | 0.9909 | 0.8478 | 0.9553 | 13 (original) |
| ANN (MLP) | 0.9893 | 0.8561 | 1.0362 | 13 (original) |
| Voting Ensemble | 0.9906 | 0.8689 | 0.9727 | 82 (trees) |

> **Note on Adjusted R²:** The test set has n=21 samples. For tree models trained on 82 features, the Adjusted R² denominator (n − p − 1) is negative, making the formula mathematically undefined. This is reported explicitly as `N/A (n≤p)` throughout the notebook. Linear models use the original 13 features, where Adjusted R² is valid and well-behaved.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ultrasonic-water-level-prediction.git
cd ultrasonic-water-level-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook Ultrasonic_water_level_FINAL_v3.ipynb
```

Run all cells from top to bottom. The notebook is fully self-contained and reproducible.

### 4. Use the deployment function

```python
import joblib, json
import pandas as pd

model    = joblib.load("water_volume_model_final.pkl")
metadata = json.load(open("water_volume_model_metadata.json"))

# Provide your sensor readings
readings = {
    "freq_khz":              40,
    "L_wave_peak_amp_water": 0.87,
    # ... all 82 feature values
}

input_df = pd.DataFrame([readings], columns=metadata["features"])
volume_L = model.predict(input_df)[0]
print(f"Predicted water volume: {volume_L:.3f} L")
```

---

## 📦 Dataset

**File:** `Features.csv`  
**Rows:** 144 (105 after IQR cleaning)  
**Columns:** 14 (13 features + 1 target)

| Column | Description |
|--------|-------------|
| `freq_khz` | Ultrasonic frequency (kHz) |
| `L_wave_peak_amp_water` | L-wave peak amplitude in water |
| `L_wave_time_ms_water` | L-wave time of flight in water (ms) |
| `T_wave_peak_amp_water` | T-wave peak amplitude in water |
| `T_wave_time_ms_water` | T-wave time of flight in water (ms) |
| `L_wave_peakamp_air` | L-wave peak amplitude in air |
| `L_wave_time_air_ms` | L-wave time of flight in air (ms) |
| `T_wave_peakamp_air` | T-wave peak amplitude in air |
| `T_wave_time_air_ms` | T-wave time of flight in air (ms) |
| `L_Wave_RF` | L-wave reflection factor |
| `T_wave_RF` | T-wave reflection factor |
| `dtof_L_wave` | Delta time-of-flight (L-wave) |
| `dtof_T_wave` | Delta time-of-flight (T-wave) |
| `volume_l` | **Target** — Water volume in litres |

---

## 🔬 Statistical Validation

- **Residual Normality:** Shapiro-Wilk p = 0.30, D'Agostino p = 0.53 → residuals are normally distributed ✅
- **Systematic Bias:** t-test p = 0.26 → mean residual is NOT significantly different from zero ✅
- **Generalisation:** Train-Validation gap = 0.0082 → no overfitting ✅
- **Prediction Intervals:** 95% empirical coverage (conservative — intervals are safely wide) ✅

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| scikit-learn | 1.8.0 | ML models, preprocessing, evaluation |
| numpy | 2.4.2 | Numerical computation |
| pandas | 3.0.1 | Data manipulation |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical visualisation |
| scipy | 1.17.0 | Statistical tests |
| joblib | 1.5.3 | Model serialisation |
| shap | latest | Model explainability |

---

## 👤 Author

**Final Year Project**  
Department of [Your Department]  
[Your University]  
[Academic Year]

---

## 📄 License

This project is submitted as an academic final year project. All rights reserved.
