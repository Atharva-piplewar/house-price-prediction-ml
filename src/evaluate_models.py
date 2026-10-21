"""
Model Evaluation and Visualization Module
Academic Project: House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models

This module generates publication-quality evaluation figures:
1. Model Comparison: MAE Bar Chart (figures/model_mae_comparison.png)
2. Model Comparison: RMSE Bar Chart (figures/model_rmse_comparison.png)
3. Model Comparison: R² Bar Chart (figures/model_r2_comparison.png)
4. Actual vs. Predicted Price Scatterplot for Best Model (figures/actual_vs_predicted.png)
5. Residual Distribution Histogram + KDE (figures/residual_distribution.png)
6. Residuals vs. Predicted Values Homoscedasticity Plot (figures/residuals_vs_predicted.png)
7. Tree-Based Feature Importance Ranking (figures/feature_importance.png)
8. Cross-Validation Stability Comparison (figures/cv_comparison.png)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import prepare_train_test_data

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Academic styling settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def generate_evaluation_figures():
    """
    Loads saved models, evaluation metrics, and test dataset to generate all 8 figures.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Load results and metadata
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    cv_path = os.path.join(RESULTS_DIR, "cv_scores.csv")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    if not os.path.exists(metrics_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Trained model artifacts or metrics not found. Run train_models.py first.")
        
    metrics_df = pd.read_csv(metrics_path)
    cv_df = pd.read_csv(cv_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    best_model_name = metadata["best_model"]
    feature_names = metadata["transformed_feature_names"]

    # 2. Load test data
    X_train, X_test, y_train, y_test, preprocessor, _, _ = prepare_train_test_data(save_artifacts=False)

    # 3. Load all models and generate predictions
    predictions = {}
    models_dict = {}
    for idx, row in metrics_df.iterrows():
        m_name = row["Model"]
        m_file = os.path.join(MODELS_DIR, row["Model_File"])
        model = joblib.load(m_file)
        models_dict[m_name] = model
        predictions[m_name] = model.predict(X_test)

    best_pred = predictions[best_model_name]
    best_residuals = y_test.values - best_pred

    palette = ["#2b5c8f", "#3a86c8", "#5ca0d3", "#82b8de", "#add0ea"]

    # -------------------------------------------------------------
    # Figure 1: MAE Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    bars = plt.bar(metrics_df["Model"], metrics_df["MAE"], color="#2b5c8f", width=0.55, edgecolor="#1a3b5c")
    plt.title("Model Comparison: Mean Absolute Error (MAE)", pad=14, fontweight="bold")
    plt.xlabel("Regression Algorithm")
    plt.ylabel("MAE in USD ($)")
    plt.xticks(rotation=15)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 400, f"${height:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="semibold")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "model_mae_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: RMSE Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    bars = plt.bar(metrics_df["Model"], metrics_df["RMSE"], color="#c0392b", width=0.55, edgecolor="#78231a")
    plt.title("Model Comparison: Root Mean Squared Error (RMSE)", pad=14, fontweight="bold")
    plt.xlabel("Regression Algorithm")
    plt.ylabel("RMSE in USD ($)")
    plt.xticks(rotation=15)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 600, f"${height:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="semibold")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "model_rmse_comparison.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: R² Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    bars = plt.bar(metrics_df["Model"], metrics_df["R2"], color="#27ae60", width=0.55, edgecolor="#196f3d")
    plt.title("Model Comparison: Coefficient of Determination (R² Score)", pad=14, fontweight="bold")
    plt.xlabel("Regression Algorithm")
    plt.ylabel("R² Score (Max: 1.0)")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.4f}", ha="center", va="bottom", fontsize=9, fontweight="semibold")
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "model_r2_comparison.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Actual vs Predicted Scatterplot for Best Model
    # -------------------------------------------------------------
    plt.figure(figsize=(7.5, 6))
    plt.scatter(y_test, best_pred, alpha=0.65, color="#2980b9", edgecolors="none", s=45, label="Hold-out Test Observations")
    min_val = min(y_test.min(), best_pred.min())
    max_val = max(y_test.max(), best_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#c0392b", linestyle="--", linewidth=2, label="Ideal 45° Line (y = ŷ)")
    plt.title(f"Actual vs. Predicted SalePrice ({best_model_name})", pad=12, fontweight="bold")
    plt.xlabel("Actual SalePrice ($ USD)")
    plt.ylabel("Predicted SalePrice ($ USD)")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Residual Distribution Histogram + KDE
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(best_residuals, kde=True, color="#2c3e50", bins=30, stat="density", linewidth=0.5)
    plt.axvline(0, color="#c0392b", linestyle="--", linewidth=1.8, label="Zero Error Reference")
    mean_err = np.mean(best_residuals)
    plt.axvline(mean_err, color="#27ae60", linestyle=":", linewidth=1.8, label=f"Mean Error: ${mean_err:,.0f}")
    plt.title(f"Residual Error Distribution ({best_model_name})", pad=12, fontweight="bold")
    plt.xlabel("Residual Error (Actual - Predicted) in $ USD")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "residual_distribution.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig5_path}")

    # -------------------------------------------------------------
    # Figure 6: Residuals vs Predicted (Homoscedasticity Check)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(best_pred, best_residuals, alpha=0.6, color="#8e44ad", edgecolors="none", s=40)
    plt.axhline(0, color="#c0392b", linestyle="--", linewidth=1.8, label="Zero Error Line")
    plt.title(f"Residuals vs. Fitted Values ({best_model_name})", pad=12, fontweight="bold")
    plt.xlabel("Fitted (Predicted) SalePrice in $ USD")
    plt.ylabel("Residual Error in $ USD")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "residuals_vs_predicted.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig6_path}")

    # -------------------------------------------------------------
    # Figure 7: Feature Importance Plot (Tree Model: XGBoost / Random Forest)
    # -------------------------------------------------------------
    tree_model = models_dict.get("XGBoost") or models_dict.get("Random Forest")
    tree_name = "XGBoost" if "XGBoost" in models_dict else "Random Forest"
    if hasattr(tree_model, "feature_importances_"):
        importances = tree_model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(9, 6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp_df, color="#16a085")
        plt.title(f"Top 15 Predictor Features by Relative Importance ({tree_name})", pad=12, fontweight="bold")
        plt.xlabel("Relative Feature Importance Weight")
        plt.ylabel("Feature")
        plt.tight_layout()
        fig7_path = os.path.join(FIGURES_DIR, "feature_importance.png")
        plt.savefig(fig7_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved: {fig7_path}")

    # -------------------------------------------------------------
    # Figure 8: 5-Fold Cross-Validation Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    x_positions = np.arange(len(cv_df))
    plt.bar(
        x_positions,
        cv_df["CV_RMSE_Mean"],
        yerr=cv_df["CV_RMSE_Std"],
        capsize=6,
        color="#34495e",
        edgecolor="#1c2833",
        width=0.55
    )
    plt.xticks(x_positions, cv_df["Model"], rotation=15)
    plt.title("5-Fold Cross-Validation Performance (Mean RMSE ± 1 Std. Dev.)", pad=14, fontweight="bold")
    plt.xlabel("Regression Algorithm")
    plt.ylabel("Cross-Validation RMSE ($ USD)")
    for i, row in cv_df.iterrows():
        plt.text(i, row["CV_RMSE_Mean"] + row["CV_RMSE_Std"] + 700, f"${row['CV_RMSE_Mean']:,.0f}", ha="center", fontsize=9, fontweight="semibold")
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "cv_comparison.png")
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {fig8_path}")

    print("\n[ALL FIGURES SUCCESSFULLY GENERATED]")


if __name__ == "__main__":
    generate_evaluation_figures()
