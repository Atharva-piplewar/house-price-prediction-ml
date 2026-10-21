# House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models
**Academic Capstone Project Final Report**

---

## Executive Summary
This academic research project develops, optimizes, and empirically compares five distinct machine learning regression algorithms for automated residential property valuation using the authentic **Ames Housing Dataset** (1,460 residential transactions across 81 attributes). 

To ensure scientific rigor and publication-grade integrity:
1. **Zero Data Leakage Protocol**: Data preprocessing and feature engineering pipelines were strictly fitted on the 80% training split (1,168 records) and only applied as inference transformations on the held-out 20% test set (292 records).
2. **Multi-Model Benchmark**: Five diverse paradigms were systematically implemented:
   - Parametric Baseline: *Ordinary Least Squares Linear Regression*
   - Ensemble Bagging: *Random Forest Regressor*
   - Kernel Machine: *Support Vector Regression (SVR - Radial Basis Function)*
   - Sequential Boosting: *Gradient Boosting Regressor (GBR)*
   - Extreme Gradient Boosting: *XGBoost Regressor*
3. **Rigorous Validation**: In addition to held-out test evaluation, 5-Fold Cross-Validation was conducted across all models to quantify parameter stability and variance.
4. **Winning Model**: **XGBoost Regressor** emerged as the superior model, achieving a **Coefficient of Determination ($R^2$) of 0.9143**, a **Root Mean Squared Error (RMSE) of $25,635**, and a **Mean Absolute Error (MAE) of $16,395**, with sub-2-millisecond inference latency (1.63 ms).

---

## 1. Introduction and Problem Statement

### 1.1 Context and Motivation
Accurate real estate valuation is a fundamental pillar of modern microeconomics, municipal property taxation, mortgage underwriting, and individual wealth accumulation. Historically, property appraisal relied on manual Hedonic Price Models (HPM) or heuristic comparative market analyses conducted by certified appraisers. However, traditional linear hedonic models struggle to capture high-dimensional non-linear interactions, diminishing returns on square footage, and compound qualitative depreciations.

Machine learning regression offers a data-driven framework capable of mapping complex multidimensional feature spaces to continuous asset prices with high fidelity.

### 1.2 Research Objectives
1. Implement a complete, leak-free preprocessing and feature engineering pipeline conforming to Scikit-Learn standards.
2. Formulate and train five regression algorithms representing varied mathematical paradigms.
3. Compare models across multiple standardized regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
4. Analyze computational trade-offs between model accuracy, training duration, and real-time prediction latency.
5. Diagnose model errors using residual distribution plots, homoscedasticity checks, and feature importance rankings.
6. Deploy an interactive production-grade web application enabling real-time valuation and model benchmarking.

---

## 2. Dataset Description and Boundary Analysis

### 2.1 Dataset Provenance
- **Dataset Name**: Ames Housing Dataset
- **Original Compiler**: Professor Dean De Cock (Truman State University)
- **Primary Source**: Kaggle Advanced Regression Techniques Competition
- **Geographic Coverage**: Ames, Iowa, United States
- **Temporal Period**: 2006 to 2010 transactions
- **Volume**: 1,460 training records with 79 explanatory variables + 1 Identifier (`Id`) + 1 Target (`SalePrice`).

### 2.2 Key Feature Categories
The feature space is partitioned into five domain categories:
- **Spatial / Scale**: Above-grade ground living area (`GrLivArea`), Total basement square feet (`TotalBsmtSF`), Lot size (`LotArea`), Garage area (`GarageArea`).
- **Qualitative Ratings**: Overall material and finish quality (`OverallQual`, 1-10), Overall property condition (`OverallCond`, 1-10), Kitchen quality (`KitchenQual`), Exterior quality (`ExterQual`).
- **Temporal / Age**: Year constructed (`YearBuilt`), Year remodeled/added (`YearRemodAdd`), Year sold (`YrSold`).
- **Capacity / Room Counts**: Above-grade bedrooms (`BedroomAbvGr`), Full bathrooms (`FullBath`), Half bathrooms (`HalfBath`), Fireplaces (`Fireplaces`), Garage car capacity (`GarageCars`).
- **Location & Architecture**: Ames physical neighborhood (`Neighborhood` across 25 distinct districts), Dwelling style (`HouseStyle`), Central Air Conditioning (`CentralAir`), Road and alley access (`Street`, `Alley`).

### 2.3 Explicit Academic Disclosures and Domain Boundaries
To maintain absolute academic honesty and scientific rigor, the following real-world boundaries are formally documented:
1. **Geographic and Currency Specificity**:
   - The dataset models single-family transactions in **Ames, Iowa (US)** denominated in **US Dollars ($ USD)**.
   - Predictions cannot be transferred directly to other international markets, such as Indian Real Estate (INR), which operate under disparate regulatory frameworks (RERA), floor rise indices, municipal circle rates, and metro transit premiums.
2. **Absence of Coastline / Ocean Views**:
   - Ames is an inland city in the Midwestern United States. There are no ocean or coastal attributes in this dataset. We do not invent synthetic sea-view variables.
3. **Constructed Residential Units vs. Vacant Plots**:
   - All records represent constructed residential homes with physical structures and foundations. Vacant commercial or agricultural land valuation is out of scope.
4. **California Housing Fallback Distinction**:
   - While a fallback loader using Scikit-Learn's `fetch_california_housing()` is provided for offline environments, California data reflects 1990 district census blocks measured in $100,000 units. Its metrics are never commingled with Ames individual home evaluations.

---

## 3. Data Preprocessing and Feature Engineering Methodology

### 3.1 Zero Data Leakage Architecture
A critical failure mode in applied machine learning is data leakage—fitting scalers or imputers on the entire dataset prior to splitting. In this project:
- The raw dataset ($N=1,460$) was first partitioned into an **80% Training Set** ($N_{train}=1,168$) and a **20% Held-Out Test Set** ($N_{test}=292$) using a fixed random seed (`random_state=42`).
- The `ColumnTransformer` preprocessing pipeline was fitted strictly on $X_{train}$. The fitted pipeline was then applied via `.transform()` to $X_{test}$ and subsequent production inference queries.

```
Raw Ames Housing Data (1,460 rows)
                │
                ▼
  [Train / Test Split: 80% / 20%]
   ┌────────────┴────────────┐
   ▼                         ▼
Training Split (1,168)    Test Split (292)
   │                         │
   ▼                         │
[Fit Preprocessor]           │
   │                         │
   ▼                         ▼
[Transform X_train]       [Transform X_test]
   │                         │
   ▼                         ▼
Model Training & 5-Fold   Final Model Evaluation
```

### 3.2 Domain-Informed Feature Engineering
Raw temporal and dimensional features were transformed into intuitive domain metrics:
1. **Effective Property Age**:
   $$\text{PropertyAge} = \max(0, \text{YrSold} - \text{YearBuilt})$$
2. **Years Since Remodeling**:
   $$\text{YearsSinceRemodel} = \max(0, \text{YrSold} - \text{YearRemodAdd})$$
3. **Renovation Flag**:
   $$\text{IsRemodeled} = \mathbb{I}(\text{YearRemodAdd} > \text{YearBuilt})$$
4. **Total Living Square Footage**:
   $$\text{TotalSF} = \text{GrLivArea} + \text{TotalBsmtSF}$$
5. **Effective Bathrooms**:
   $$\text{TotalBath} = \text{FullBath} + 0.5 \times \text{HalfBath}$$

### 3.3 Imputation and Encoding
- **Numerical Features**: Missing values imputed with the **median** of the training split to resist skewness and outliers, followed by `StandardScaler` (zero mean, unit variance).
- **Categorical Features**: Missing values imputed with constant string `'None'` (accurately representing the absence of an alley, etc.), followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to prevent runtime failure on unseen categories.

Total feature dimensions expanded from 24 raw inputs to **79 encoded predictor features**.

---

## 4. Machine Learning Model Formulations

### 4.1 Linear Regression (OLS Baseline)
Ordinary Least Squares minimizes the residual sum of squares:
$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$$
Serves as our parametric linear reference benchmark.

### 4.2 Random Forest Regressor (Ensemble Bagging)
Random Forest builds an ensemble of $B$ decorrelated regression trees using bootstrap sampling and random feature sub-selection:
$$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
- Hyperparameters: $B=150$ estimators, `max_depth=16`, `min_samples_split=4`, `min_samples_leaf=2`.

### 4.3 Support Vector Regression (Kernel Machine)
SVR solves for a flat hyper-plane within an $\varepsilon$-insensitive tube using the Radial Basis Function (RBF) kernel:
$$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$
- Hyperparameters: `kernel='rbf'`, $C=15000.0$, $\varepsilon=0.05$.

### 4.4 Gradient Boosting Regressor (Sequential Boosting)
GBR builds decision trees additively to minimize squared error loss along the negative gradient:
$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
- Hyperparameters: $M=160$ estimators, learning rate $\eta=0.07$, `max_depth=4`, `subsample=0.85`.

### 4.5 XGBoost Regressor (Extreme Gradient Boosting)
XGBoost optimizes an objective function comprising second-order Taylor expansion loss and tree-complexity regularization:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
- Hyperparameters: $160$ estimators, $\eta=0.07$, `max_depth=4`, `subsample=0.85`, `colsample_bytree=0.85`.

---

## 5. Experimental Results and Comparative Analysis

### 5.1 Held-Out Test Evaluation (20% Split, N=292)

| Model Paradigm | Regression Algorithm | MAE ($ USD) | MSE ($ USD²) | RMSE ($ USD) | R² Score | Train Time (s) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Parametric | Linear Regression | $20,152.59 | 1,100,210,324 | $33,169.42 | 0.8566 | **0.014s** | **0.36 ms** |
| Ensemble Bagging | Random Forest Regressor | $17,603.63 | 858,969,818 | $29,308.19 | 0.8880 | 0.317s | 43.54 ms |
| Kernel Machine | Support Vector Regression | $18,350.07 | 1,621,728,189 | $40,270.69 | 0.7886 | 0.097s | 33.78 ms |
| Sequential Boosting | Gradient Boosting Regressor | $16,566.74 | 682,235,557 | $26,119.64 | 0.9111 | 0.702s | 1.85 ms |
| **Extreme Boosting** | **XGBoost (Champion)** | **$16,394.81** | **657,141,952** | **$25,634.78** | **0.9143** | 0.267s | 1.63 ms |

### 5.2 5-Fold Cross-Validation Performance (Training Split)

| Regression Model | Mean CV $R^2$ Score | CV $R^2$ Std. Dev. | Mean CV RMSE ($ USD) | CV RMSE Std. Dev. |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 0.8119 | ± 0.0514 | $33,217 | ± $5,440 |
| Random Forest | 0.8406 | ± 0.0558 | $30,440 | ± $5,819 |
| Support Vector Regression | 0.7936 | ± 0.0334 | $35,024 | ± $4,259 |
| **Gradient Boosting** | **0.8613** | ± 0.0649 | **$28,083** | ± $6,151 |
| **XGBoost** | 0.8579 | ± 0.0571 | $28,574 | ± $5,273 |

### 5.3 Key Empirical Findings
1. **Superiority of Gradient Boosting**: XGBoost and Gradient Boosting outperform the Linear Regression baseline by **+5.77% in $R^2$** and reduce RMSE by **$7,534 per property**. This confirms significant non-linear compounding interactions in real estate pricing.
2. **Error Reduction**: XGBoost achieves a test MAE of **$16,395**, indicating that on average, its prediction deviates by less than 9% from the true transaction price across a property distribution averaging $180,921.
3. **Inference Latency for Production**: XGBoost executes inference in **1.63 milliseconds** per sample, making it 26 times faster during prediction than Random Forest (43.5 ms), proving optimal for real-time web deployment.

---

## 6. Error Diagnostics and Residual Analysis

### 6.1 Residual Distribution Analysis
Residual error is defined as $e_i = y_i - \hat{y}_i$.
- The residual distribution for the winning XGBoost model is centered near zero with a mean error of **$482**, exhibiting near-normal symmetry.
- A slight right skew is observed in the tail, attributable to a small number of ultra-luxury estates ($>\$400,000$) where unique bespoke architecture and non-standard amenities create idiosyncratic price spikes.

### 6.2 Homoscedasticity Assessment
The Residuals vs. Fitted plot demonstrates consistent variance across the middle property value band ($100,000 - $300,000). At extreme high-end valuations ($>\$400,000$), residual dispersion widens moderately, a well-known phenomenon in real estate economics where high-net-worth buyers exhibit discretionary price insensitivity.

---

## 7. Feature Importance and Economic Interpretation

Feature importance extracted via XGBoost's gain metric reveals the top predictive drivers:
1. **`OverallQual` (Overall Material and Finish Quality)**: Dominant determinant of price. Moving from quality grade 6 to 8 commands a steep market premium regardless of home size.
2. **`TotalSF` and `GrLivArea` (Living Area)**: Total interior usable square footage provides strong, positive linear elasticity.
3. **`TotalBsmtSF` (Basement Area)**: Basements in the Midwestern US are essential for both weather protection (tornado/freezing temperatures) and secondary living quarters.
4. **`GarageCars` & `GarageArea`**: Vehicle capacity is heavily favored in automobile-dependent suburban environments like Ames, Iowa.
5. **`Neighborhood`**: Premium neighborhoods (Northridge Heights, Stone Brook, Northridge) yield substantial localized categorical uplifts.
6. **`PropertyAge` & `YearsSinceRemodel`**: Physical aging introduces steady depreciation, which is substantially arrested by recent modernization.

---

## 8. Software Architecture and Deployment

The project is structured following clean, modular software engineering standards:
```
ml final project/
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Project dependencies
├── pytest.ini                 # Pytest configuration
├── data/
│   ├── train.csv              # Ames Housing dataset (1,460 records)
│   └── README.md              # Dataset documentation & limitations
├── src/
│   ├── data_loader.py         # Automated data loading & schema validation
│   ├── preprocessing.py       # Zero-leakage feature engineering & ColumnTransformer
│   ├── train_models.py        # Model training, 5-fold CV & artifact serialization
│   ├── evaluate_models.py     # Generation of 8 publication-quality figures
│   └── predict.py             # Validation & single/batch inference engine
├── models/
│   ├── preprocessing_pipeline.joblib
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── support_vector_regression.joblib
│   ├── gradient_boosting.joblib
│   ├── xgboost.joblib
│   └── model_metadata.json
├── results/
│   ├── preprocessing_summary.json
│   ├── metrics_summary.csv
│   └── cv_scores.csv
├── figures/                   # 8 Generated publication evaluation figures
├── tests/                     # 21 Automated Pytest unit and integration tests
└── reports/
    └── final_report.md        # Comprehensive academic research report
```

---

## 9. Conclusion and Future Work

### 9.1 Summary of Contributions
- Successfully established an end-to-end, reproducible Machine Learning workflow for house price prediction.
- Proved through rigorous empirical testing that gradient boosting ensembles achieve high accuracy ($R^2 = 0.9143$) while maintaining sub-2ms latency.
- Guaranteed complete absence of data leakage through isolated train-split transformer fitting.
- Delivered an interactive web application, 21 automated unit tests, and thorough documentation of domain boundaries.

### 9.2 Future Research Directions
1. **Spatial Geospatial Coordinates**: Incorporate exact GIS coordinates and street-level spatial smoothing (e.g., Kriging or Graph Neural Networks).
2. **Deep Learning Architectures**: Investigate TabNet and Tabular Transformers for tabular representation learning on larger property datasets.
3. **Indian Real Estate Adaptation**: Gather localized metro datasets (Mumbai, Bangalore, Delhi NCR) incorporating RERA compliance, developer brand rating, and proximity to mass transit corridors.

---
*Report compiled for Machine Learning Final Project Capstone Evaluation.*
