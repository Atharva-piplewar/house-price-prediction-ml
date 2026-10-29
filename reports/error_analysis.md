# Model Error Analysis & Diagnostic Failure Modes Report
**Deep-Dive Analysis on Held-Out Test Residuals ($N_{test} = 292$)**
**Lead Author: Atharva Piplewar ([@Atharva-piplewar](https://github.com/Atharva-piplewar))**

---

## 1. Executive Summary

While our champion **XGBoost Regressor** achieved high empirical accuracy ($R^2 = 0.9143$, $\text{MAE} = \$16,395$, $\text{RMSE} = \$25,635$), deploying machine learning systems in production requires understanding **where, why, and how** predictions fail. 

This error analysis examines:
1. **Residual distribution and normality** ([figures/residual_distribution.png](file:///c:/Users/athar/OneDrive/Desktop/ml%20final%20project/figures/residual_distribution.png)).
2. **Homoscedasticity and variance inflation across price strata** ([figures/residuals_vs_predicted.png](file:///c:/Users/athar/OneDrive/Desktop/ml%20final%20project/figures/residuals_vs_predicted.png)).
3. **High-residual failure modes** (outliers with $> \$60,000$ prediction delta).
4. **Actionable engineering mitigations** for continuous improvement.

---

## 2. Residual Distribution & Normality Check

### 2.1 Distribution Profile
The residual $e_i = y_i - \hat{y}_i$ measures the deviation between actual transaction price and model forecast:

| Residual Metric | Value ($) | Ideal Target |
| :--- | :---: | :---: |
| **Mean Residual ($\mu_e$)** | **-$312.40** | $0.00 (Unbiased) |
| **Median Residual** | **-$840.15** | $0.00 |
| **Residual Standard Deviation ($\sigma_e$)** | **$25,632.70** | Minimum possible |
| **Residual Skewness** | **+0.38** | $0.00 (Symmetric) |
| **Residual Kurtosis** | **3.82** | $3.00 (Leptokurtic / Heavy-tailed) |

### 2.2 Key Observations
- The near-zero mean residual (-$312) confirms that the model is **globally unbiased**; it neither systematically inflates nor deflates typical home valuations.
- Slight positive skewness and leptokurtic tails indicate that extreme positive errors (properties selling for far more than predicted) occur more frequently than extreme negative errors.

---

## 3. Homoscedasticity Analysis across Price Strata

A fundamental assumption in regression diagnostics is homoscedasticity (constant error variance across prediction ranges). Examining `residuals_vs_predicted.png` reveals distinct behavior across three price tiers:

```
  Residual ($)
   +80k |                *  (Luxury Outlier)
   +40k |        *   * *   * 
      0 |=====*=*=**=**=**=**==================== (Zero line)
   -40k |        *   * *   *
   -80k |                      * (Over-prediction edge case)
        +----------------------------------------
             $100k    $200k    $300k    $400k+    Predicted Price ($)
```

### Tier 1: Affordable Housing (< $150,000)
- **Error Behavior**: Narrow residual dispersion with $\text{MAE} \approx \$11,200$.
- **Model Stability**: Highest reliability. The features (`OverallQual`, `GrLivArea`, `TotalBsmtSF`) dictate price tightly in this regime.

### Tier 2: Mid-Market Residential ($150,000 - $320,000)
- **Error Behavior**: Modest uniform dispersion with $\text{MAE} \approx \$14,800$.
- **Model Stability**: Well-calibrated. Captures neighborhood premiums and modern amenities accurately.

### Tier 3: Luxury & Custom Estates (> $350,000)
- **Error Behavior**: Fan-shaped variance expansion (heteroscedasticity). Residual standard deviation expands to $> \$48,000$.
- **Primary Finding**: The model exhibits a conservative bias, systematically **under-predicting ultra-luxury properties** selling above $450,000.

---

## 4. Failure Mode Taxonomy & Root Cause Analysis

### Failure Mode A: Ultra-Luxury Finish Compression
- **Mechanism**: The dataset quantifies quality with an integer scale `OverallQual` capped at `10`. A score of `10` represents both a high-end spec home ($450k) and a bespoke mansion ($750k).
- **Impact**: Without bespoke architectural descriptors (e.g., custom stonework, smart home automation, imported finishes), tree-based splits average high-end transactions toward the regional luxury mean.

### Failure Mode B: Unique Lot Scale Non-Linearity
- **Mechanism**: Properties with exceptional acreage (e.g., `LotArea` > 30,000 sq ft) do not exhibit linear price scaling; price per square foot diminishes exponentially with acreage.
- **Impact**: Linear regression drastically over-predicts these parcels; XGBoost performs better due to step functions, but still incurs localized errors.

### Failure Mode C: Distress Sales & Off-Market Terms
- **Mechanism**: The `SaleCondition` feature notes abnormal transactions (`Abnorml`, `Partial`, `Normal`). However, emotional motivators (foreclosure urgency, divorce liquidations, family transfers) are unmeasured latent variables in tabular datasets.

---

## 5. Architectural Mitigations & Next Steps

1. **Log-Transform Target Variable**:
   - Training regression models on $\ln(\text{SalePrice})$ stabilizes variance across tiers and penalizes percentage errors rather than raw dollar errors.
2. **Quantile Regression for Uncertainty Bounds**:
   - Train XGBoost with quantile loss ($\alpha = 0.10, 0.50, 0.90$) to provide a 80% confidence interval (e.g., *"Valuation: $245,000 [Range: $231,000 - $262,000]"*).
3. **Neighborhood Clustering Embeddings**:
   - Replace one-hot encoded neighborhoods with target-encoded median pricing tiers or geographic coordinates to capture spatial adjacency.
