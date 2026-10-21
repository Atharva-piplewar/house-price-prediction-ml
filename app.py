"""
Streamlit Web Application: House Price Prediction and Comparative Regression Analysis
Academic Project: House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models
Dataset: Ames Housing Dataset (1,460 Residential Homes, 81 Features)
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure system path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_ames_housing
from src.predict import predict_house_price, get_default_property_input, load_model_metadata

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Streamlit Page Setup
st.set_page_config(
    page_title="House Price Prediction & Comparative ML Analysis",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }
    
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    
    .badge-item {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    .badge-item-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .badge-item-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        margin: 1.5rem 0;
    }
    
    .prediction-label {
        color: #a5b4fc;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    .prediction-meta {
        color: #c7d2fe;
        font-size: 0.95rem;
    }
    
    .spec-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .spec-title {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .spec-value {
        color: #f1f5f9;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .info-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #cbd5e1;
    }
    
    .boundary-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444;
        padding: 1.2rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_metrics():
    """Loads metrics summary and CV scores CSV files."""
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    cv_path = os.path.join(RESULTS_DIR, "cv_scores.csv")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    cv_df = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
    
    with open(meta_path, "r") as f:
        meta = json.load(f) if os.path.exists(meta_path) else {}
        
    return metrics_df, cv_df, meta


@st.cache_data
def load_cached_raw_data():
    """Loads raw Ames Housing dataset for EDA."""
    return load_ames_housing()


# Load metadata and metrics
metrics_df, cv_df, meta = load_all_metrics()
raw_data = load_cached_raw_data()

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏡 House Price Prediction & Comparative ML Analysis</div>
    <div class="hero-subtitle">
        An end-to-end Machine Learning research benchmark evaluating five regression algorithms on the authentic 
        Ames Housing dataset (1,460 residential properties, 81 attributes) with zero data leakage, rigorous 5-fold cross-validation, 
        and real-time inference.
    </div>
    <div class="badge-container">
        <span class="badge-item-success">🏆 Best Model: XGBoost (R²: 0.9143)</span>
        <span class="badge-item">🎯 Test RMSE: $25,635</span>
        <span class="badge-item">📉 Test MAE: $16,395</span>
        <span class="badge-item">📊 5 Models Evaluated</span>
        <span class="badge-item">🛡️ 0% Data Leakage Guarantee</span>
        <span class="badge-item-warning">📍 Ames, Iowa (USD $)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_calc, tab_benchmark, tab_importance, tab_eda, tab_docs = st.tabs([
    "🎯 Live Valuation Calculator",
    "📊 Model Benchmark & Comparison",
    "🔍 Feature Importance & Drivers",
    "📈 Exploratory Data Analysis (EDA)",
    "📜 Academic Documentation & Bounds"
])

# ==============================================================================
# TAB 1: LIVE VALUATION CALCULATOR
# ==============================================================================
with tab_calc:
    st.subheader("Interactive House Valuation Engine")
    st.markdown(
        "Adjust structural, architectural, and temporal property characteristics below to estimate the "
        "fair market value. Predictions are generated in real-time through the production preprocessing pipeline."
    )
    
    col_preset, col_model_choice = st.columns([1, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Quick Property Presets:",
            ["Standard Median Home", "Spacious Suburban Family Home", "Compact Vintage Home", "Modern High-End Estate"]
        )
    with col_model_choice:
        model_selection = st.selectbox(
            "Select Inference Algorithm:",
            [
                "Auto Best Model (XGBoost)",
                "Compare All 5 Models Simultaneously",
                "XGBoost",
                "Gradient Boosting",
                "Random Forest",
                "Linear Regression",
                "Support Vector Regression"
            ]
        )

    # Populate preset values
    default_vals = get_default_property_input()
    if preset_choice == "Spacious Suburban Family Home":
        default_vals.update({
            "GrLivArea": 2400, "LotArea": 12500, "OverallQual": 8, "OverallCond": 6,
            "YearBuilt": 2008, "YearRemodAdd": 2012, "TotalBsmtSF": 1300,
            "BedroomAbvGr": 4, "FullBath": 2, "HalfBath": 1, "GarageCars": 3,
            "GarageArea": 780, "Fireplaces": 2, "HouseStyle": "2Story",
            "KitchenQual": "Gd", "ExterQual": "Gd", "Neighborhood": "NridgHt"
        })
    elif preset_choice == "Compact Vintage Home":
        default_vals.update({
            "GrLivArea": 1050, "LotArea": 6000, "OverallQual": 5, "OverallCond": 6,
            "YearBuilt": 1945, "YearRemodAdd": 1970, "TotalBsmtSF": 750,
            "BedroomAbvGr": 2, "FullBath": 1, "HalfBath": 0, "GarageCars": 1,
            "GarageArea": 280, "Fireplaces": 0, "HouseStyle": "1Story",
            "KitchenQual": "TA", "ExterQual": "TA", "Neighborhood": "OldTown"
        })
    elif preset_choice == "Modern High-End Estate":
        default_vals.update({
            "GrLivArea": 3400, "LotArea": 17000, "OverallQual": 10, "OverallCond": 7,
            "YearBuilt": 2018, "YearRemodAdd": 2020, "TotalBsmtSF": 1800,
            "BedroomAbvGr": 5, "FullBath": 3, "HalfBath": 2, "GarageCars": 3,
            "GarageArea": 920, "Fireplaces": 3, "HouseStyle": "2Story",
            "KitchenQual": "Ex", "ExterQual": "Ex", "Neighborhood": "NoRidge"
        })

    # Input Form
    with st.form("valuation_form"):
        st.markdown("#### 1. Dimensions and Structural Specifications")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gr_liv_area = st.number_input("Above-Ground Living Area (sq ft):", min_value=300, max_value=8000, value=int(default_vals["GrLivArea"]), step=50)
            total_bsmt_sf = st.number_input("Basement Area (sq ft):", min_value=0, max_value=6000, value=int(default_vals["TotalBsmtSF"]), step=50)
        with c2:
            lot_area = st.number_input("Lot Size (sq ft):", min_value=1000, max_value=100000, value=int(default_vals["LotArea"]), step=250)
            garage_area = st.number_input("Garage Size (sq ft):", min_value=0, max_value=2000, value=int(default_vals["GarageArea"]), step=25)
        with c3:
            bedrooms = st.number_input("Bedrooms Above Grade:", min_value=1, max_value=8, value=int(default_vals["BedroomAbvGr"]))
            full_bath = st.number_input("Full Bathrooms:", min_value=0, max_value=6, value=int(default_vals["FullBath"]))
        with c4:
            half_bath = st.number_input("Half Bathrooms:", min_value=0, max_value=4, value=int(default_vals["HalfBath"]))
            garage_cars = st.number_input("Garage Car Capacity:", min_value=0, max_value=5, value=int(default_vals["GarageCars"]))

        st.markdown("#### 2. Quality, Condition, and Age")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            overall_qual = st.slider("Overall Quality (1 = Poor, 10 = Very Excellent):", min_value=1, max_value=10, value=int(default_vals["OverallQual"]))
        with q2:
            overall_cond = st.slider("Overall Condition (1 = Poor, 10 = Very Excellent):", min_value=1, max_value=10, value=int(default_vals["OverallCond"]))
        with q3:
            year_built = st.number_input("Year Built:", min_value=1870, max_value=2026, value=int(default_vals["YearBuilt"]))
        with q4:
            year_remod = st.number_input("Year Remodeled / Added:", min_value=int(year_built), max_value=2026, value=max(int(year_built), int(default_vals["YearRemodAdd"])))

        st.markdown("#### 3. Architectural, Style, and Neighborhood Attributes")
        s1, s2, s3, s4 = st.columns(4)
        
        neighborhood_list = [
            'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
            'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
            'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
            'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
        ]
        house_styles = ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin']
        quality_grades = ['Ex', 'Gd', 'TA', 'Fa']
        
        with s1:
            neighborhood = st.selectbox("Ames Neighborhood:", neighborhood_list, index=neighborhood_list.index(default_vals["Neighborhood"]) if default_vals["Neighborhood"] in neighborhood_list else 0)
            house_style = st.selectbox("House Style:", house_styles, index=house_styles.index(default_vals["HouseStyle"]) if default_vals["HouseStyle"] in house_styles else 0)
        with s2:
            kitchen_qual = st.selectbox("Kitchen Quality:", quality_grades, index=quality_grades.index(default_vals["KitchenQual"]) if default_vals["KitchenQual"] in quality_grades else 1, format_func=lambda x: {"Ex": "Excellent (Ex)", "Gd": "Good (Gd)", "TA": "Average / Typical (TA)", "Fa": "Fair (Fa)"}.get(x, x))
            exter_qual = st.selectbox("Exterior Material Quality:", quality_grades, index=quality_grades.index(default_vals["ExterQual"]) if default_vals["ExterQual"] in quality_grades else 1, format_func=lambda x: {"Ex": "Excellent (Ex)", "Gd": "Good (Gd)", "TA": "Average / Typical (TA)", "Fa": "Fair (Fa)"}.get(x, x))
        with s3:
            central_air = st.selectbox("Central Air Conditioning:", ["Y", "N"], index=0 if default_vals["CentralAir"] == "Y" else 1, format_func=lambda x: "Yes" if x == "Y" else "No")
            fireplaces = st.number_input("Number of Fireplaces:", min_value=0, max_value=4, value=int(default_vals["Fireplaces"]))
        with s4:
            lot_config = st.selectbox("Lot Configuration:", ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], index=0)
            land_contour = st.selectbox("Land Contour:", ['Lvl', 'Bnk', 'HLS', 'Low'], format_func=lambda x: {"Lvl": "Near Flat/Level", "Bnk": "Banked", "HLS": "Hillside", "Low": "Depression"}.get(x, x))

        submitted = st.form_submit_button("⚡ Calculate Property Valuation", use_container_width=True, type="primary")

    # Property dictionary
    property_input = {
        "GrLivArea": gr_liv_area,
        "LotArea": lot_area,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "YrSold": 2026,
        "TotalBsmtSF": total_bsmt_sf,
        "BedroomAbvGr": bedrooms,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "Fireplaces": fireplaces,
        "HouseStyle": house_style,
        "KitchenQual": kitchen_qual,
        "ExterQual": exter_qual,
        "Neighborhood": neighborhood,
        "Street": "Pave",
        "Alley": "None",
        "Condition1": "Norm",
        "LandContour": land_contour,
        "LotConfig": lot_config,
        "CentralAir": central_air,
    }

    # Execute Prediction
    try:
        if model_selection == "Compare All 5 Models Simultaneously":
            st.markdown("### 🏆 Multi-Model Valuation Comparison")
            all_models = ["Linear Regression", "Random Forest", "Support Vector Regression", "Gradient Boosting", "XGBoost"]
            comparison_results = []
            
            for m_name in all_models:
                t0 = time.perf_counter()
                res = predict_house_price(property_input, model_name=m_name)
                lat = (time.perf_counter() - t0) * 1000
                comparison_results.append({
                    "Algorithm": m_name,
                    "Estimated Valuation (USD)": f"${res['predicted_price_usd']:,.2f}",
                    "Raw Price": res['predicted_price_usd'],
                    "Inference Latency (ms)": round(lat, 2)
                })
            
            comp_df = pd.DataFrame(comparison_results)
            
            # Highlight Best Model
            best_m_row = comp_df[comp_df["Algorithm"] == "XGBoost"].iloc[0]
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Primary Valuation Recommendation (XGBoost - Best Algorithm)</div>
                <div class="prediction-value">{best_m_row['Estimated Valuation (USD)']}</div>
                <div class="prediction-meta">Expected Accuracy Band (± RMSE): <b>${best_m_row['Raw Price'] - 25635:,.0f} &mdash; ${best_m_row['Raw Price'] + 25635:,.0f} USD</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(comp_df[["Algorithm", "Estimated Valuation (USD)", "Inference Latency (ms)"]], use_container_width=True)
            
            # Interactive Chart for Comparison
            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444"]
            bars = ax.barh(comp_df["Algorithm"], comp_df["Raw Price"], color=colors, height=0.55)
            ax.set_xlabel("Predicted Property Valuation ($ USD)")
            ax.set_title("Cross-Model Price Estimates for Given Property", fontweight="bold", pad=12)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 2000, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(0, max(comp_df["Raw Price"]) * 1.15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:
            selected_m = None if "Auto" in model_selection else model_selection
            res = predict_house_price(property_input, model_name=selected_m)
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Fair Market Valuation ({res['model_used']})</div>
                <div class="prediction-value">${res['predicted_price_usd']:,.2f}</div>
                <div class="prediction-meta">Denominated in <b>US Dollars (USD)</b> &bull; Ames, Iowa Housing Market Calibration</div>
            </div>
            """, unsafe_allow_html=True)

        # Derived Feature Breakdown
        st.markdown("#### 📐 Derived Living & Structural Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="spec-card">
                <div class="spec-title">Total Living Area (TotalSF)</div>
                <div class="spec-value">{gr_liv_area + total_bsmt_sf:,.0f} sq ft</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="spec-card">
                <div class="spec-title">Property Age</div>
                <div class="spec-value">{2026 - year_built} years</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="spec-card">
                <div class="spec-title">Total Bathrooms</div>
                <div class="spec-value">{full_bath + 0.5 * half_bath:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            is_remod = "Yes" if year_remod > year_built else "Original"
            st.markdown(f"""
            <div class="spec-card">
                <div class="spec-title">Remodeled Status</div>
                <div class="spec-value">{is_remod}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==============================================================================
# TAB 2: MODEL BENCHMARK & COMPARISON DASHBOARD
# ==============================================================================
with tab_benchmark:
    st.subheader("Model Performance Benchmark & Comparative Analysis")
    st.markdown(
        "Five distinct regression paradigms were evaluated strictly on a held-out 20% test split (292 records) "
        "and validated using 5-fold cross validation. All transformations were fitted solely on the 80% training set."
    )
    
    # KPI metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("🏆 Best Model", "XGBoost", "RMSE: $25,635")
    with m2:
        st.metric("📈 Test R² Score", "0.9143", "+0.0577 vs Baseline")
    with m3:
        st.metric("📉 Test MAE", "$16,395", "Lowest Average Error")
    with m4:
        st.metric("⚡ Inference Speed", "1.63 ms", "Real-Time Production")
    with m5:
        st.metric("🛡️ CV R² (Mean)", "0.8579", "5-Fold Validated")

    st.markdown("---")
    st.markdown("### 📋 Held-Out Test Set Performance Table")
    if metrics_df is not None:
        styled_df = metrics_df.copy()
        styled_df["MAE"] = styled_df["MAE"].apply(lambda x: f"${x:,.2f}")
        styled_df["MSE"] = styled_df["MSE"].apply(lambda x: f"{x:,.0f}")
        styled_df["RMSE"] = styled_df["RMSE"].apply(lambda x: f"${x:,.2f}")
        styled_df["Training_Time_Sec"] = styled_df["Training_Time_Sec"].apply(lambda x: f"{x:.4f}s")
        styled_df["Prediction_Time_Sec"] = styled_df["Prediction_Time_Sec"].apply(lambda x: f"{x*1000:.2f} ms")
        st.dataframe(styled_df[["Model", "MAE", "RMSE", "R2", "Training_Time_Sec", "Prediction_Time_Sec"]], use_container_width=True)

    st.markdown("### 🔄 5-Fold Cross-Validation Performance (Training Split)")
    if cv_df is not None:
        cv_disp = cv_df.copy()
        cv_disp["CV_RMSE_Mean"] = cv_disp["CV_RMSE_Mean"].apply(lambda x: f"${x:,.2f}")
        cv_disp["CV_RMSE_Std"] = cv_disp["CV_RMSE_Std"].apply(lambda x: f"± ${x:,.2f}")
        cv_disp["CV_R2_Mean"] = cv_disp["CV_R2_Mean"].apply(lambda x: f"{x:.4f}")
        cv_disp["CV_R2_Std"] = cv_disp["CV_R2_Std"].apply(lambda x: f"± {x:.4f}")
        st.dataframe(cv_disp[["Model", "CV_R2_Mean", "CV_R2_Std", "CV_RMSE_Mean", "CV_RMSE_Std"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖼️ Publication Evaluation Figures")
    fig_selection = st.radio(
        "Select Figure to Inspect:",
        [
            "Model Error Comparison (MAE & RMSE)",
            "Model R² Comparison",
            "Actual vs. Predicted Scatterplot (Best Model)",
            "Residual Distribution & Homoscedasticity Diagnostics",
            "5-Fold Cross Validation Stability",
        ],
        horizontal=True
    )

    if fig_selection == "Model Error Comparison (MAE & RMSE)":
        c_mae, c_rmse = st.columns(2)
        with c_mae:
            st.image(os.path.join(FIGURES_DIR, "model_mae_comparison.png"), caption="Figure 1: Mean Absolute Error (MAE) Comparison across Models")
        with c_rmse:
            st.image(os.path.join(FIGURES_DIR, "model_rmse_comparison.png"), caption="Figure 2: Root Mean Squared Error (RMSE) Comparison across Models")
    elif fig_selection == "Model R² Comparison":
        st.image(os.path.join(FIGURES_DIR, "model_r2_comparison.png"), caption="Figure 3: Coefficient of Determination (R²) across Models", width=750)
    elif fig_selection == "Actual vs. Predicted Scatterplot (Best Model)":
        st.image(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), caption="Figure 4: Actual vs Predicted SalePrice Scatterplot (XGBoost)", width=650)
    elif fig_selection == "Residual Distribution & Homoscedasticity Diagnostics":
        r1, r2 = st.columns(2)
        with r1:
            st.image(os.path.join(FIGURES_DIR, "residual_distribution.png"), caption="Figure 5: Residual Error Distribution with Zero-Mean Reference")
        with r2:
            st.image(os.path.join(FIGURES_DIR, "residuals_vs_predicted.png"), caption="Figure 6: Residuals vs Fitted Values (Homoscedasticity Check)")
    elif fig_selection == "5-Fold Cross Validation Stability":
        st.image(os.path.join(FIGURES_DIR, "cv_comparison.png"), caption="Figure 8: 5-Fold Cross-Validation Performance (Mean RMSE ± 1 Std. Dev.)", width=750)

# ==============================================================================
# TAB 3: FEATURE IMPORTANCE & DRIVERS
# ==============================================================================
with tab_importance:
    st.subheader("Feature Importance & Real Estate Valuation Drivers")
    st.markdown(
        "Understanding which attributes most strongly influence valuation is critical for homeowners, "
        "appraisers, and financial underwriters. The plot below highlights the top 15 features by relative gain "
        "weight computed by the winning XGBoost model."
    )
    
    st.image(os.path.join(FIGURES_DIR, "feature_importance.png"), caption="Figure 7: Top 15 Predictor Features Ranked by Importance (XGBoost)", use_container_width=True)
    
    st.markdown("### 💡 Key Empirical Insights:")
    st.markdown("""
    1. **Overall Construction Quality (`OverallQual`)**: Consistently the single strongest linear and non-linear predictor. A jump from quality score 6 to 8 commands a steep market premium regardless of square footage.
    2. **Total Living Area (`TotalSF` / `GrLivArea`)**: Combined above-ground and finished basement square footage exhibits a powerful positive elasticity on home valuation.
    3. **Basement Square Footage (`TotalBsmtSF`)**: Properties with large full basements provide substantial utility value in Ames, Iowa due to Midwestern climatic conditions (storm protection and insulated usable space).
    4. **Neighborhood Location Premia**: Specific neighborhoods such as **Northridge Heights (`NridgHt`)**, **Stone Brook (`StoneBr`)**, and **Northridge (`NoRidge`)** introduce distinct categorical uplifts in valuation.
    5. **Property Age (`PropertyAge` & `YearsSinceRemodel`)**: Physical depreciation negatively impacts price, while recent renovations effectively reset the functional age of the structure.
    """)

# ==============================================================================
# TAB 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab_eda:
    st.subheader("Exploratory Data Analysis: Ames Housing Dataset")
    st.markdown(
        "Ames Housing dataset contains **1,460 residential transaction records** with **81 original attributes** "
        "compiled by Prof. Dean De Cock. Below is an exploratory inspection of the underlying dataset."
    )
    
    e1, e2, e3 = st.columns(3)
    with e1:
        st.metric("Total Records", f"{len(raw_data):,}")
    with e2:
        st.metric("Median Sale Price", f"${raw_data['SalePrice'].median():,.0f}")
    with e3:
        st.metric("Mean Sale Price", f"${raw_data['SalePrice'].mean():,.0f}")

    st.markdown("#### 1. Dataset Sample Preview")
    preview_cols = ["Id", "SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt", "Neighborhood", "HouseStyle"]
    st.dataframe(raw_data[preview_cols].head(10), use_container_width=True)

    st.markdown("#### 2. Target Variable (`SalePrice`) Distribution")
    fig_eda, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Histogram
    sns.histplot(raw_data["SalePrice"], kde=True, color="#2563eb", ax=ax_hist, bins=35)
    ax_hist.set_title("SalePrice Distribution (Right-Skewed)", fontweight="bold")
    ax_hist.set_xlabel("SalePrice ($ USD)")
    ax_hist.axvline(raw_data["SalePrice"].median(), color="red", linestyle="--", label=f"Median: ${raw_data['SalePrice'].median():,.0f}")
    ax_hist.legend()
    
    # Boxplot by Overall Quality
    sns.boxplot(x="OverallQual", y="SalePrice", data=raw_data, palette="Blues_r", ax=ax_box)
    ax_box.set_title("SalePrice Distribution across Overall Quality (1-10)", fontweight="bold")
    ax_box.set_xlabel("Overall Quality Rating")
    ax_box.set_ylabel("SalePrice ($ USD)")
    
    plt.tight_layout()
    st.pyplot(fig_eda)
    plt.close()

    st.markdown("#### 3. Top Numerical Feature Correlations with SalePrice")
    num_cols_eda = ["SalePrice", "GrLivArea", "TotalBsmtSF", "OverallQual", "GarageCars", "GarageArea", "YearBuilt", "YearRemodAdd", "FullBath", "Fireplaces"]
    corr_matrix = raw_data[num_cols_eda].corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax_corr, cbar=True, square=True)
    ax_corr.set_title("Correlation Heatmap: Top Predictors vs. SalePrice", fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close()

# ==============================================================================
# TAB 5: ACADEMIC DOCUMENTATION & BOUNDS
# ==============================================================================
with tab_docs:
    st.subheader("Academic Documentation, Methodology, and Project Scope")
    
    st.markdown("""
    ### 1. Research Objectives and Architecture
    This project investigates machine learning regression techniques for residential real estate valuation.
    The primary objectives are:
    - **Parametric vs. Non-Parametric Modeling**: Benchmark classical Linear Regression against Ensemble Bagging (Random Forest), Kernelized Methods (SVR), and Sequential Gradient Boosting (GBR & XGBoost).
    - **Elimination of Data Leakage**: Employ strict Scikit-Learn `ColumnTransformer` pipelines where missing value imputation (median for numerical, constant for categorical) and standardization are fitted solely on the 80% training split.
    - **Generalization Assessment**: Evaluate models through nested 5-Fold Cross-Validation on the training set and verify on an untouched 20% hold-out test set.
    """)

    st.markdown("""
    <div class="boundary-box">
        <h4 style="color: #b91c1c; margin-top: 0;">⚠️ Crucial Academic Disclosures and Domain Boundaries</h4>
        <ol style="color: #475569; margin-bottom: 0;">
            <li><b>Currency and Geographic Context</b>: 
                The Ames Housing dataset reflects real estate dynamics in Ames, Iowa (United States) between 2006 and 2010. 
                All valuations are expressed in <b>US Dollars ($ USD)</b>. These models cannot be directly applied to Indian Real Estate (INR), 
                which is governed by distinct local determinants (e.g., RERA certification, floor rise index, municipal circle rates, metro proximity).
            </li>
            <li><b>Inland Midwestern Geography (No Coastline/Ocean View)</b>: 
                Ames, Iowa is situated in the inland Upper Midwest. There are <b>no ocean, sea, or coastal attributes</b> in this dataset. 
                Rather than fabricating synthetic coastline parameters, we document this intrinsic geographic boundary.
            </li>
            <li><b>Constructed Residential Dwellings vs. Vacant Land</b>: 
                All records represent single-family or attached residential dwellings with foundations and structures. 
                Vacant agricultural or commercial plots are not modeled.
            </li>
            <li><b>California Housing Fallback Distinction</b>: 
                While the repository provides a fallback loader using <code>fetch_california_housing()</code> for offline/restricted environments, 
                California housing captures 1990 census block groups with target values in units of $100,000 and district-level metrics. 
                California metrics are never commingled with Ames Housing results.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 2. Experimental Results Summary
    | Model Paradigm | Algorithm | Test MAE ($) | Test RMSE ($) | Test R² Score | Training Time (s) | Inference Latency |
    | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
    | **Baseline Parametric** | Linear Regression | $20,153 | $33,169 | 0.8566 | 0.014s | 0.36 ms |
    | **Ensemble Bagging** | Random Forest Regressor | $17,604 | $29,308 | 0.8880 | 0.317s | 43.54 ms |
    | **Kernel Machine** | Support Vector Regression (RBF) | $18,350 | $40,271 | 0.7886 | 0.097s | 33.78 ms |
    | **Sequential Boosting**| Gradient Boosting Regressor | $16,567 | $26,120 | 0.9111 | 0.702s | 1.85 ms |
    | **Extreme Gradient Boosting** | **XGBoost (Champion)** | **$16,395** | **$25,635** | **0.9143** | **0.267s** | **1.63 ms** |

    ### 3. Key Conclusions
    - **XGBoost achieved the superior predictive accuracy** with an R² of **0.9143** and a Root Mean Squared Error of **$25,635**, explaining 91.4% of the variance in held-out property values.
    - **Gradient Boosting and XGBoost demonstrated clear superiority** over classical Linear Regression (+5.7% R² improvement), highlighting the non-linear interaction effects present in residential real estate (e.g., quality ratings modifying square footage value).
    - **Inference latency** for XGBoost is **1.63 ms**, making it exceptionally suitable for production deployment in web applications and automated valuation models (AVMs).
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;'>"
    "Academic Machine Learning Capstone Project &bull; House Price Prediction & Comparative Regression Analysis &bull; Ames Housing Dataset"
    "</div>",
    unsafe_allow_html=True
)
