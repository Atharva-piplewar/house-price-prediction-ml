# Model Card: Extreme Gradient Boosting (XGBoost) for House Price Valuation

Following the Model Cards for Model Reporting framework ([Mitchell et al., 2019](https://arxiv.org/abs/1810.03993)).

---

## 1. Model Details

- **Model Name**: Ames Residential Valuation Engine (XGBoost Regressor)
- **Model Version**: `1.0.0`
- **Model Type**: Extreme Gradient Boosting Regressor (Tree-based ensemble)
- **Developer / Author**: Atharva Piplewar ([@Atharva-piplewar](https://github.com/Atharva-piplewar))
- **Date Developed**: October 2026
- **Release License**: MIT License
- **Framework & Dependencies**:
  - Python >= 3.11
  - XGBoost >= 2.0.0
  - Scikit-Learn >= 1.3.0
  - Joblib >= 1.3.0
- **Serialization Artifacts**:
  - Model weights: `models/xgboost.joblib` (258 KB)
  - Preprocessing pipeline: `models/preprocessing_pipeline.joblib` (7 KB)
  - Metadata record: `models/model_metadata.json`

### 1.1 Hyperparameters
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_estimators` | `150` | Balanced tree capacity avoiding overfitting |
| `learning_rate` | `0.05` | Conservative shrinkage preventing gradient oscillation |
| `max_depth` | `4` | Restricts tree depth to 4th-order feature interactions |
| `subsample` | `0.8` | Row stochastic subsampling reduces variance |
| `colsample_bytree`| `0.8` | Column subsampling decorrelates individual trees |
| `objective` | `reg:squarederror` | Standard squared L2 loss for continuous pricing |
| `random_state` | `42` | Deterministic reproducibility guarantee |

---

## 2. Intended Use & Boundaries

### 2.1 Primary Intended Uses
- Automated real-estate preliminary valuation (AVM) for single-family residential homes in Ames, Iowa.
- Rapid comparative property screening for buyers, realtors, and institutional property evaluators.
- Sensitivity analysis for property renovations (estimating added value of finished basement area, renovated kitchen, or garage expansion).

### 2.2 Out-of-Scope & Prohibited Applications
- **Predatory Mortgage Underwriting**: The model should not be used as the sole automated decision-maker for denying mortgage loans without human appraisal oversight.
- **Commercial & Industrial Real Estate**: The model was trained exclusively on single-family residential properties and will produce invalid appraisals on commercial parcels.
- **Geographic Extrapolation**: The model must not be applied directly to non-Ames metropolitan areas (e.g., San Francisco, New York, Mumbai) without local recalibration and transfer learning.

---

## 3. Data & Preprocessing Pipeline

### 3.1 Training & Held-Out Test Splits
- **Source Dataset**: Ames Housing Dataset (compiled by Dean De Cock, Truman State University).
- **Split Ratio**: 80% Training ($N_{train} = 1,168$) / 20% Held-Out Test ($N_{test} = 292$).
- **Stratification**: Random seed `42` preserved across all baseline comparisons.

### 3.2 Zero-Leakage Pipeline Architecture
To eliminate data leakage, all feature scalers and median imputers were strictly fitted on $N_{train}$ only:
1. **Numeric Pipeline**:
   - `SimpleImputer(strategy='median')`: Robust against extreme spatial outliers.
   - `StandardScaler()`: Zero mean and unit variance normalization.
2. **Categorical Pipeline**:
   - `SimpleImputer(strategy='constant', fill_value='Missing')`: Preserves structural absence flags (e.g., No Alley, No Basement).
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Encodes nominal features into dummy indicators safely ignoring unseen levels at inference.
3. **Engineered Domain Features**:
   - `TotalSF = GrLivArea + TotalBsmtSF` (Combined living area)
   - `HouseAge = YrSold - YearBuilt` (Effective building age at sale)
   - `RemodAge = YrSold - YearRemodAdd` (Time since last major remodel)
   - `TotalBaths = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`

---

## 4. Quantitative Evaluation Metrics

All models were evaluated on the identical held-out test split ($N = 292$ records) and cross-validated across 5 folds:

### 4.1 Comparative Test Benchmark
| Model | MAE ($) | RMSE ($) | $R^2$ Score | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor (Champion)** | **$16,394.81** | **$25,634.78** | **0.9143** | **0.264 s** | **1.40 ms** |
| Gradient Boosting Regressor | $16,566.74 | $26,119.64 | 0.9111 | 0.638 s | 2.23 ms |
| Random Forest Regressor | $17,603.63 | $29,308.19 | 0.8880 | 0.338 s | 44.07 ms |
| Linear Regression | $20,152.59 | $33,169.42 | 0.8566 | 0.014 s | 0.51 ms |
| Support Vector Regression (SVR) | $18,350.07 | $40,270.69 | 0.7886 | 0.075 s | 30.94 ms |

### 4.2 5-Fold Cross-Validation Stability
- **Mean CV $R^2$**: $0.8579 \pm 0.0571$
- **Mean CV RMSE**: $\$28,574 \pm \$5,273$
- Demonstrates consistent generalization across varied data partitions without catastrophic fold collapse.

---

## 5. Explainability & Feature Drivers

Top features driving XGBoost predictions (Gini impurity / Gain ranking):
1. **`OverallQual`** (Overall material and finish rating): Primary valuation anchor.
2. **`GrLivArea`** (Above-grade ground living area): Main linear scale predictor.
3. **`TotalBsmtSF`** (Total basement square footage): Strong living space multiplier.
4. **`GarageCars` / `GarageArea`**: Vehicle amenity indicator.
5. **`YearBuilt` / `HouseAge`**: Depreciation and era value.
6. **`Neighborhood` (Northridge Heights, Stone Brook, Somerset)**: Location premium factor.

---

## 6. Environmental & Computational Footprint

- **Hardware Platform**: Consumer x86-64 CPU (AMD / Intel multi-core)
- **Training Duration**: $0.264\text{ seconds}$
- **Inference Latency**: $1.40\text{ milliseconds per transaction}$
- **Estimated Carbon Footprint**: $< 0.0001\text{ kg CO}_2\text{e}$ (negligible green compute profile)

---

## 7. Model Maintenance & Monitoring Guidelines

- **Recalibration Frequency**: Annual retraining with updated municipal tax assessments and transaction records.
- **Data Drift Triggers**: Monitor median `SalePrice` and distribution shifts in `GrLivArea` using Population Stability Index (PSI).
- **Fallback Mechanism**: Automated fallbacks are integrated into `src/data_loader.py` and `src/predict.py` with input boundary clipping.
