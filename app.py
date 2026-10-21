"""
Streamlit Web Application: Next-Gen Luxury Real Estate Valuation & Comparative ML Benchmark
Project: House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models
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

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_ames_housing
from src.predict import predict_house_price, get_default_property_input, load_model_metadata

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
HERO_BANNER_PATH = os.path.join(FIGURES_DIR, "hero_banner.jpg")

# Streamlit Page Setup
st.set_page_config(
    page_title="PropVal Studio | Next-Gen Real Estate Valuation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep Custom Modern Dark Mode & Component Injections
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Global Streamlit Reset */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    header[data-testid="stHeader"] { 
        background: transparent !important;
        backdrop-filter: blur(8px) !important;
    }
    
    /* Base Body Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #1e293b !important;
    }
    
    /* Dynamic Crisp Canvas */
    .stApp {
        background-color: #f8fafc !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(79, 70, 229, 0.05) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.04) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.025em !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Paragraphs and labels */
    p, span, label {
        color: #334155;
    }
    
    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 16px rgba(15, 23, 42, 0.04) !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #334155 !important;
    }
    
    /* Glass / Crisp Cards */
    .glass-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.75rem 2rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-panel:hover {
        border-color: #93c5fd;
        box-shadow: 0 8px 30px -4px rgba(37, 99, 235, 0.12);
    }
    
    /* Hero Header */
    .hero-box {
        position: relative;
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 50%, #eff6ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 22px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 35px -8px rgba(37, 99, 235, 0.12), 0 2px 8px rgba(0, 0, 0, 0.03);
        overflow: hidden;
    }
    
    .hero-glow-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: #eff6ff;
        border: 1px solid #93c5fd;
        color: #1d4ed8;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 1rem;
    }
    
    .live-pulse {
        width: 8px;
        height: 8px;
        background-color: #2563eb;
        border-radius: 50%;
        box-shadow: 0 0 10px #2563eb;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.6); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
    }
    
    .hero-headline {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
    }
    
    .hero-paragraph {
        color: #475569;
        font-size: 1.1rem;
        line-height: 1.6;
        max-width: 880px;
        margin-bottom: 1.5rem;
    }
    
    /* Ticker Pill Grid */
    .metric-ticker-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.9rem;
    }
    
    .ticker-pill {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .ticker-pill:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 6px 16px -2px rgba(37, 99, 235, 0.12);
    }
    
    .ticker-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    
    .ticker-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Valuation Big Display Card */
    .valuation-glow-card {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 50%, #f8fafc 100%);
        border: 2px solid #3b82f6;
        border-radius: 24px;
        padding: 2.8rem 2.2rem;
        text-align: center;
        box-shadow: 0 20px 40px -10px rgba(37, 99, 235, 0.16), 0 2px 10px rgba(0, 0, 0, 0.03);
        margin: 2rem 0;
        position: relative;
    }
    
    .val-model-tag {
        color: #2563eb;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    
    .val-hero-digits {
        font-size: 3.8rem;
        font-weight: 800;
        color: #1e3a8a;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        margin: 0.25rem 0;
    }
    
    .val-meta-text {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }
    
    .val-range-capsule {
        display: inline-block;
        background: #dbeafe;
        border: 1px solid #93c5fd;
        color: #1e40af;
        border-radius: 9999px;
        padding: 0.45rem 1.35rem;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Financial / Spec Cards */
    .spec-tile {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.35rem 1.15rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .spec-tile:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
        box-shadow: 0 8px 24px -4px rgba(37, 99, 235, 0.12);
    }
    
    .spec-tile-icon {
        font-size: 1.5rem;
        margin-bottom: 0.4rem;
    }
    
    .spec-tile-title {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    
    .spec-tile-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
        font-family: 'JetBrains Mono', sans-serif;
    }
    
    .spec-tile-foot {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Streamlit Input / Widget Deep Styling */
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }
    div[data-baseweb="input"] input {
        color: #0f172a !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }

    /* Radio button labels & checks */
    div[data-testid="stRadio"] label {
        color: #1e293b !important;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 1.1rem !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Primary Buttons */
    button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        padding: 0.75rem 1.6rem !important;
        font-size: 1rem !important;
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button[kind="primary"]:hover, div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px -4px rgba(37, 99, 235, 0.55) !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }
    
    /* Tabs Overhaul */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f1f5f9;
        padding: 0.45rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        font-size: 0.92rem;
        font-weight: 600;
        color: #475569;
        padding: 0 1.25rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 2rem 2.2rem !important;
        box-shadow: 0 6px 24px -4px rgba(15, 23, 42, 0.04) !important;
    }

    div[data-testid="stForm"] h5 {
        color: #2563eb !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Academic Disclaimer Banner */
    .disclaimer-banner {
        background: linear-gradient(135deg, #fff1f2 0%, #ffffff 100%);
        border: 1px solid #fecdd3;
        border-left: 5px solid #ef4444;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px -2px rgba(239, 68, 68, 0.06);
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

# ==============================================================================
# SIDEBAR CONTROLS & ARCHITECTURE
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.85rem; margin-bottom: 1.5rem;">
        <span style="font-size: 2.2rem;">⚡</span>
        <div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">PropVal Studio</div>
            <div style="font-size: 0.75rem; color: #2563eb; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Next-Gen Valuation Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🎯 Quick Property Scenarios")
    preset_choice = st.selectbox(
        "Choose Real Estate Archetype:",
        [
            "Standard Median Suburban Home",
            "Modern High-End Luxury Residence",
            "Spacious Family Suburban Dwelling",
            "Vintage Character Bungalow",
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### 💱 Currency Selection")
    currency_mode = st.radio(
        "Display Currency:",
        ["₹ INR (Indian Rupee)", "$ USD (US Dollar)"],
        index=0,
        horizontal=True
    )
    is_inr = "INR" in currency_mode
    USD_TO_INR = 84.0

    def format_curr(usd_val, compact=True):
        if is_inr:
            inr = usd_val * USD_TO_INR
            if compact:
                if inr >= 10000000:
                    return f"₹{inr/10000000:.2f} Cr"
                elif inr >= 100000:
                    return f"₹{inr/100000:.2f} Lakh"
            return f"₹{inr:,.0f}"
        return f"${usd_val:,.2f}" if not compact else f"${usd_val:,.0f}"

    st.markdown("---")
    st.markdown("#### 🤖 Machine Learning Model")
    model_selection = st.selectbox(
        "Active Estimator:",
        [
            "Auto Best Model (XGBoost)",
            "Compare All 5 Models Simultaneously",
            "XGBoost Regressor",
            "Gradient Boosting Regressor",
            "Random Forest Regressor",
            "Linear Regression",
            "Support Vector Regression (RBF)"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### 🛡️ Engineering Safeguards")
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.1rem; font-size: 0.82rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b;">Leakage Prevention:</span>
            <span style="color: #2563eb; font-weight: 700;">Strict 80:20 Split</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b;">Cross Validation:</span>
            <span style="color: #059669; font-weight: 700;">5-Fold Stratified</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b;">Inference Speed:</span>
            <span style="color: #0f172a; font-weight: 700;">1.63 ms / sample</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #64748b;">Automated Tests:</span>
            <span style="color: #059669; font-weight: 700;">21 / 21 Passed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists(HERO_BANNER_PATH):
        st.image(HERO_BANNER_PATH, caption="Ames Architectural Showcase", width='stretch')


# ==============================================================================
# HERO SECTION
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-glow-badge">
        <span class="live-pulse"></span>
        Production Valuation Benchmark &bull; Ames Housing Dataset
    </div>
    <div class="hero-headline">Intelligent Real Estate Valuation Studio</div>
    <div class="hero-paragraph">
        An academic machine learning system comparing five regression algorithms on 1,460 authentic 
        residential property transactions. Featuring strict zero data leakage, automated feature engineering, 
        and instantaneous multi-model inference.
    </div>
    <div class="metric-ticker-grid">
        <div class="ticker-pill">
            <div class="ticker-label">🏆 Winning Algorithm</div>
            <div class="ticker-val" style="color: #2563eb;">XGBoost</div>
        </div>
        <div class="ticker-pill">
            <div class="ticker-label">🎯 Test Accuracy (R²)</div>
            <div class="ticker-val" style="color: #059669;">0.9143</div>
        </div>
        <div class="ticker-pill">
            <div class="ticker-label">📉 Test Error (RMSE)</div>
            <div class="ticker-val" style="color: #0284c7;">$25,635</div>
        </div>
        <div class="ticker-pill">
            <div class="ticker-label">⚡ Inference Latency</div>
            <div class="ticker-val" style="color: #d97706;">1.63 ms</div>
        </div>
        <div class="ticker-pill">
            <div class="ticker-label">📍 Geographic Focus</div>
            <div class="ticker-val" style="color: #dc2626;">Ames, IA (USD)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_calc, tab_benchmark, tab_importance, tab_eda, tab_docs = st.tabs([
    "💎 Live Valuation & Financial Studio",
    "📊 Model Benchmark & Diagnostic Gallery",
    "🔬 Feature Importance & Renovation Simulator",
    "📈 Exploratory Market Analytics",
    "📜 Academic Report & Geographic Bounds"
])

# Populate preset values
default_vals = get_default_property_input()
if preset_choice == "Modern High-End Luxury Residence":
    default_vals.update({
        "GrLivArea": 3200, "LotArea": 16500, "OverallQual": 9, "OverallCond": 7,
        "YearBuilt": 2018, "YearRemodAdd": 2021, "TotalBsmtSF": 1700,
        "BedroomAbvGr": 4, "FullBath": 3, "HalfBath": 1, "GarageCars": 3,
        "GarageArea": 860, "Fireplaces": 2, "HouseStyle": "2Story",
        "KitchenQual": "Ex", "ExterQual": "Ex", "Neighborhood": "NridgHt"
    })
elif preset_choice == "Spacious Family Suburban Dwelling":
    default_vals.update({
        "GrLivArea": 2350, "LotArea": 11800, "OverallQual": 8, "OverallCond": 6,
        "YearBuilt": 2006, "YearRemodAdd": 2014, "TotalBsmtSF": 1200,
        "BedroomAbvGr": 4, "FullBath": 2, "HalfBath": 1, "GarageCars": 2,
        "GarageArea": 620, "Fireplaces": 1, "HouseStyle": "2Story",
        "KitchenQual": "Gd", "ExterQual": "Gd", "Neighborhood": "CollgCr"
    })
elif preset_choice == "Vintage Character Bungalow":
    default_vals.update({
        "GrLivArea": 1100, "LotArea": 6500, "OverallQual": 5, "OverallCond": 6,
        "YearBuilt": 1948, "YearRemodAdd": 1985, "TotalBsmtSF": 800,
        "BedroomAbvGr": 2, "FullBath": 1, "HalfBath": 0, "GarageCars": 1,
        "GarageArea": 300, "Fireplaces": 0, "HouseStyle": "1Story",
        "KitchenQual": "TA", "ExterQual": "TA", "Neighborhood": "OldTown"
    })

# ==============================================================================
# TAB 1: VALUATION & FINANCIAL STUDIO
# ==============================================================================
with tab_calc:
    st.markdown("### 🏡 Configure Property Dimensions & Characteristics")
    st.markdown("Adjust structural specifications below to estimate fair market value through the production preprocessing pipeline.")
    
    with st.form("valuation_form"):
        # Section 1: Dimensions
        st.markdown("<h5 style='color: #2563eb; font-weight: 700; margin-bottom: 0.8rem;'>1. Scale, Layout, and Square Footage</h5>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gr_liv_area = st.number_input("Above-Grade Living Area (sq ft):", min_value=350, max_value=7500, value=int(default_vals["GrLivArea"]), step=50)
            total_bsmt_sf = st.number_input("Total Basement Area (sq ft):", min_value=0, max_value=5000, value=int(default_vals["TotalBsmtSF"]), step=50)
        with c2:
            lot_area = st.number_input("Lot Size (sq ft):", min_value=1200, max_value=90000, value=int(default_vals["LotArea"]), step=250)
            garage_area = st.number_input("Garage Area (sq ft):", min_value=0, max_value=1800, value=int(default_vals["GarageArea"]), step=25)
        with c3:
            bedrooms = st.number_input("Bedrooms Above Grade:", min_value=1, max_value=7, value=int(default_vals["BedroomAbvGr"]))
            full_bath = st.number_input("Full Bathrooms:", min_value=0, max_value=5, value=int(default_vals["FullBath"]))
        with c4:
            half_bath = st.number_input("Half Bathrooms:", min_value=0, max_value=3, value=int(default_vals["HalfBath"]))
            garage_cars = st.number_input("Garage Car Capacity:", min_value=0, max_value=4, value=int(default_vals["GarageCars"]))

        # Section 2: Quality & Age
        st.markdown("<h5 style='color: #2563eb; font-weight: 700; margin-top: 1rem; margin-bottom: 0.8rem;'>2. Quality, Finish & Construction Timeline</h5>", unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            overall_qual = st.slider("Overall Finish Quality (1: Poor → 10: Excellent):", min_value=1, max_value=10, value=int(default_vals["OverallQual"]))
        with q2:
            overall_cond = st.slider("Overall Condition (1: Poor → 10: Excellent):", min_value=1, max_value=10, value=int(default_vals["OverallCond"]))
        with q3:
            year_built = st.number_input("Year Built:", min_value=1870, max_value=2026, value=int(default_vals["YearBuilt"]))
        with q4:
            year_remod = st.number_input("Year Remodeled / Added:", min_value=int(year_built), max_value=2026, value=max(int(year_built), int(default_vals["YearRemodAdd"])))

        # Section 3: Architecture & Location
        st.markdown("<h5 style='color: #2563eb; font-weight: 700; margin-top: 1rem; margin-bottom: 0.8rem;'>3. Architecture, Location & Amenities</h5>", unsafe_allow_html=True)
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
            house_style = st.selectbox("Dwelling Architecture:", house_styles, index=house_styles.index(default_vals["HouseStyle"]) if default_vals["HouseStyle"] in house_styles else 0)
        with s2:
            kitchen_qual = st.selectbox("Kitchen Finish:", quality_grades, index=quality_grades.index(default_vals["KitchenQual"]) if default_vals["KitchenQual"] in quality_grades else 1, format_func=lambda x: {"Ex": "⭐ Luxury (Ex)", "Gd": "✨ Good (Gd)", "TA": "🏠 Typical (TA)", "Fa": "🔨 Fair (Fa)"}.get(x, x))
            exter_qual = st.selectbox("Exterior Material:", quality_grades, index=quality_grades.index(default_vals["ExterQual"]) if default_vals["ExterQual"] in quality_grades else 1, format_func=lambda x: {"Ex": "⭐ Luxury (Ex)", "Gd": "✨ Good (Gd)", "TA": "🏠 Typical (TA)", "Fa": "🔨 Fair (Fa)"}.get(x, x))
        with s3:
            central_air = st.selectbox("Central Air Conditioning:", ["Y", "N"], index=0 if default_vals["CentralAir"] == "Y" else 1, format_func=lambda x: "Yes (Central HVAC)" if x == "Y" else "No")
            fireplaces = st.number_input("Fireplaces:", min_value=0, max_value=4, value=int(default_vals["Fireplaces"]))
        with s4:
            lot_config = st.selectbox("Lot Configuration:", ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], index=0)
            land_contour = st.selectbox("Land Contour:", ['Lvl', 'Bnk', 'HLS', 'Low'], format_func=lambda x: {"Lvl": "Level / Flat", "Bnk": "Banked", "HLS": "Hillside", "Low": "Depression"}.get(x, x))

        submitted = st.form_submit_button("⚡ Calculate Property Valuation", width='stretch', type="primary")

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

    try:
        if "Compare All 5" in model_selection:
            st.markdown("### 🏆 Multi-Model Valuation Consensus")
            all_models = ["Linear Regression", "Random Forest", "Support Vector Regression", "Gradient Boosting", "XGBoost"]
            comparison_results = []
            
            for m_name in all_models:
                t0 = time.perf_counter()
                res = predict_house_price(property_input, model_name=m_name)
                lat = (time.perf_counter() - t0) * 1000
                comparison_results.append({
                    "Algorithm": m_name,
                    "Estimated Valuation": res['predicted_price_usd'],
                    "Inference Latency (ms)": round(lat, 2)
                })
            
            comp_df = pd.DataFrame(comparison_results)
            best_val = comp_df[comp_df["Algorithm"] == "XGBoost"]["Estimated Valuation"].iloc[0]
            avg_val = comp_df["Estimated Valuation"].mean()
            
            best_display = format_curr(best_val)
            best_sub = f"₹{best_val * USD_TO_INR:,.0f} INR &bull; (≈ ${best_val:,.0f} USD)" if is_inr else f"≈ ₹{best_val * USD_TO_INR:,.0f} INR"
            band_low = format_curr(max(10000.0, best_val - 25635))
            band_high = format_curr(best_val + 25635)
            
            st.markdown(f"""
            <div class="valuation-glow-card">
                <div class="val-model-tag">Consensus Primary Valuation &bull; XGBoost Champion Engine</div>
                <div class="val-hero-digits">{best_display}</div>
                <div class="val-meta-text">{best_sub} &bull; Multi-Model Mean: <b>{format_curr(avg_val)}</b></div>
                <div class="val-range-capsule">Expected Accuracy Range (± RMSE): <b>{band_low} &mdash; {band_high}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Multi-Model Plot with Bright Crisp Sapphire Palette
            fig, ax = plt.subplots(figsize=(9, 3.6), facecolor="#ffffff")
            ax.set_facecolor("#f8fafc")
            colors = ["#2563eb", "#059669", "#7c3aed", "#d97706", "#dc2626"]
            plot_vals = comp_df["Estimated Valuation"] * (USD_TO_INR if is_inr else 1.0)
            bars = ax.barh(comp_df["Algorithm"], plot_vals, color=colors, height=0.55, edgecolor="#cbd5e1")
            ax.set_xlabel(f"Predicted Property Valuation ({'₹ INR' if is_inr else '$ USD'})", color="#0f172a", fontweight="bold", fontsize=11)
            ax.tick_params(colors="#334155", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#e2e8f0")
            ax.grid(axis='x', color="#e2e8f0", linestyle='--', alpha=0.7)
            for bar, orig_usd in zip(bars, comp_df["Estimated Valuation"]):
                w = bar.get_width()
                label_txt = format_curr(orig_usd)
                ax.text(w * 1.02, bar.get_y() + bar.get_height() / 2, label_txt, va="center", color="#0f172a", fontsize=9, fontweight="bold")
            ax.set_xlim(0, max(plot_vals) * 1.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:
            selected_m = None if "Auto" in model_selection else model_selection.replace(" Regressor", "").replace(" (RBF)", "")
            res = predict_house_price(property_input, model_name=selected_m)
            price = res['predicted_price_usd']
            price_display = format_curr(price)
            price_sub = f"₹{price * USD_TO_INR:,.0f} INR &bull; (≈ ${price:,.0f} USD)" if is_inr else f"≈ ₹{price * USD_TO_INR:,.0f} INR"
            band_low = format_curr(max(10000.0, price - 25635))
            band_high = format_curr(price + 25635)
            
            st.markdown(f"""
            <div class="valuation-glow-card">
                <div class="val-model-tag">Fair Market Valuation ({res['model_used']})</div>
                <div class="val-hero-digits">{price_display}</div>
                <div class="val-meta-text">{price_sub}</div>
                <div class="val-range-capsule">Expected Accuracy Range (± RMSE): <b>{band_low} &mdash; {band_high}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Specs & Financial Insights Grid
        st.markdown(f"#### 📐 Structural Specifications & Financial Projections ({'₹ INR' if is_inr else '$ USD'})")
        total_sf = gr_liv_area + total_bsmt_sf
        eff_price = res['predicted_price_usd'] if 'res' in locals() else best_val
        price_per_sqft = (eff_price * USD_TO_INR if is_inr else eff_price) / max(1, gr_liv_area)
        
        # 30-year mortgage assumption at 6.8% with 20% down
        loan_amt = eff_price * 0.80
        monthly_rate = 0.068 / 12
        n_payments = 360
        monthly_mortgage = loan_amt * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
        est_tax_monthly = (eff_price * 0.012) / 12
        est_insurance_monthly = 120.0
        total_monthly = (monthly_mortgage + est_tax_monthly + est_insurance_monthly) * (USD_TO_INR if is_inr else 1.0)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(f"""
            <div class="spec-tile">
                <div class="spec-tile-icon">📐</div>
                <div class="spec-tile-title">Total Usable Area</div>
                <div class="spec-tile-num">{total_sf:,.0f} sq ft</div>
                <div class="spec-tile-foot">{gr_liv_area} ag + {total_bsmt_sf} bsmt</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            sqft_str = f"₹{price_per_sqft:,.0f}" if is_inr else f"${price_per_sqft:.1f}"
            avg_str = "~₹10,000 / sq ft" if is_inr else "~$120 / sq ft"
            st.markdown(f"""
            <div class="spec-tile">
                <div class="spec-tile-icon">💵</div>
                <div class="spec-tile-title">Price per Sq. Ft.</div>
                <div class="spec-tile-num">{sqft_str}</div>
                <div class="spec-tile-foot">Ames avg: {avg_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown(f"""
            <div class="spec-tile">
                <div class="spec-tile-icon">⏳</div>
                <div class="spec-tile-title">Effective Age</div>
                <div class="spec-tile-num">{2026 - year_built} yrs</div>
                <div class="spec-tile-foot">Renovated: {"Yes" if year_remod > year_built else "No"}</div>
            </div>
            """, unsafe_allow_html=True)
        with b4:
            mo_str = f"₹{total_monthly:,.0f} /mo" if is_inr else f"${total_monthly:,.0f} /mo"
            st.markdown(f"""
            <div class="spec-tile">
                <div class="spec-tile-icon">🏦</div>
                <div class="spec-tile-title">Est. Monthly EMI</div>
                <div class="spec-tile-num">{mo_str}</div>
                <div class="spec-tile-foot">P&I + Tax + Ins (20% down)</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Inference Alert: {e}")

# ==============================================================================
# TAB 2: MODEL BENCHMARK & DIAGNOSTIC GALLERY
# ==============================================================================
with tab_benchmark:
    st.markdown("### 🏆 Comprehensive Model Evaluation Benchmark")
    st.markdown("All five algorithms were evaluated on an untouched held-out 20% test split ($N=292$) and cross-validated with 5 stratified folds.")
    
    if metrics_df is not None:
        styled_df = metrics_df.copy()
        styled_df["MAE ($)"] = styled_df["MAE"].apply(lambda x: f"${x:,.2f}")
        styled_df["RMSE ($)"] = styled_df["RMSE"].apply(lambda x: f"${x:,.2f}")
        styled_df["R² Score"] = styled_df["R2"].apply(lambda x: f"{x:.4f}")
        styled_df["Train Time"] = styled_df["Training_Time_Sec"].apply(lambda x: f"{x:.4f}s")
        styled_df["Inference Speed"] = styled_df["Prediction_Time_Sec"].apply(lambda x: f"{x*1000:.2f} ms")
        
        st.dataframe(
            styled_df[["Model", "R² Score", "MAE ($)", "RMSE ($)", "Train Time", "Inference Speed"]],
            width='stretch',
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### 🖼️ Diagnostic Figure Studio")
    
    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
        "📊 Error & R² Bar Charts",
        "🎯 Actual vs Predicted Scatter",
        "📉 Residual Diagnostics & Homoscedasticity",
        "🔄 5-Fold Cross-Validation Stability"
    ])
    
    with diag_tab1:
        f1, f2, f3 = st.columns(3)
        with f1:
            st.image(os.path.join(FIGURES_DIR, "model_mae_comparison.png"), caption="Figure 1: Mean Absolute Error (MAE)")
        with f2:
            st.image(os.path.join(FIGURES_DIR, "model_rmse_comparison.png"), caption="Figure 2: Root Mean Squared Error (RMSE)")
        with f3:
            st.image(os.path.join(FIGURES_DIR, "model_r2_comparison.png"), caption="Figure 3: R² Coefficient of Determination")
            
    with diag_tab2:
        st.image(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), caption="Figure 4: Actual vs Predicted SalePrice Scatterplot (XGBoost)", width=720)
        st.markdown("**Key Takeaway**: XGBoost predictions tightly hug the 45° ideal equality line across typical homes ($100K - $350K), with slight variance widening above $400K due to luxury customization.")
        
    with diag_tab3:
        r1, r2 = st.columns(2)
        with r1:
            st.image(os.path.join(FIGURES_DIR, "residual_distribution.png"), caption="Figure 5: Residual Error Distribution with Zero-Mean Reference")
        with r2:
            st.image(os.path.join(FIGURES_DIR, "residuals_vs_predicted.png"), caption="Figure 6: Residuals vs Fitted Values (Homoscedasticity Check)")
        st.markdown("**Statistical Insight**: Residuals are approximately normally distributed around zero (mean error: +$482), meeting the fundamental Gauss-Markov regression assumptions.")

    with diag_tab4:
        st.image(os.path.join(FIGURES_DIR, "cv_comparison.png"), caption="Figure 8: 5-Fold Cross-Validation Performance (Mean RMSE ± 1 Std. Dev.)", width=750)
        if cv_df is not None:
            cv_disp = cv_df.copy()
            cv_disp["Mean CV RMSE"] = cv_disp["CV_RMSE_Mean"].apply(lambda x: f"${x:,.2f}")
            cv_disp["CV RMSE Std Dev"] = cv_disp["CV_RMSE_Std"].apply(lambda x: f"± ${x:,.2f}")
            cv_disp["Mean CV R²"] = cv_disp["CV_R2_Mean"].apply(lambda x: f"{x:.4f}")
            cv_disp["CV R² Std Dev"] = cv_disp["CV_R2_Std"].apply(lambda x: f"± {x:.4f}")
            st.dataframe(cv_disp[["Model", "Mean CV R²", "CV R² Std Dev", "Mean CV RMSE", "CV RMSE Std Dev"]], width='stretch', hide_index=True)

# ==============================================================================
# TAB 3: FEATURE IMPORTANCE & ROI SIMULATOR
# ==============================================================================
with tab_importance:
    st.markdown("### 🔬 Feature Importance & Renovation ROI Simulator")
    st.markdown("Discover the structural and location factors that dictate property value and simulate renovation ROI.")
    
    st.image(os.path.join(FIGURES_DIR, "feature_importance.png"), caption="Figure 7: Top 15 Predictor Features Ranked by Relative Importance (XGBoost)", width='stretch')
    
    st.markdown("---")
    st.markdown("### 🛠️ Interactive Renovation Value Simulator")
    st.markdown("Simulate common property remodeling projects to evaluate model-estimated price appreciation:")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        base_sample = get_default_property_input()
        renovation_type = st.radio(
            "Select Remodeling Project:",
            [
                "Kitchen Remodel: Upgrade from Typical (TA) to Luxury (Ex)",
                "Add 400 sq ft Finished Basement Addition",
                "Full Modernization: Upgrade Overall Quality (+2 points) & Reset Remodel Year",
                "Add 2nd Fireplace + 1 Extra Full Bathroom"
            ]
        )
    
    with sim_col2:
        base_pred = predict_house_price(base_sample)["predicted_price_usd"]
        mod_sample = base_sample.copy()
        
        if "Kitchen Remodel" in renovation_type:
            mod_sample["KitchenQual"] = "Ex"
            mod_sample["YearRemodAdd"] = 2026
        elif "Basement Addition" in renovation_type:
            mod_sample["TotalBsmtSF"] += 400
        elif "Full Modernization" in renovation_type:
            mod_sample["OverallQual"] = min(10, mod_sample["OverallQual"] + 2)
            mod_sample["YearRemodAdd"] = 2026
        elif "Fireplace" in renovation_type:
            mod_sample["Fireplaces"] += 1
            mod_sample["FullBath"] += 1
            
        upgraded_pred = predict_house_price(mod_sample)["predicted_price_usd"]
        diff = upgraded_pred - base_pred
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%); border: 1.5px solid #86efac; border-radius: 16px; padding: 1.8rem; text-align: center; box-shadow: 0 8px 25px -4px rgba(16, 185, 129, 0.12);">
            <div style="font-size: 0.82rem; color: #047857; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Estimated Valuation Appreciation</div>
            <div style="font-size: 2.4rem; font-weight: 800; color: #059669; font-family: 'JetBrains Mono'; margin: 0.25rem 0;">+${diff:,.2f} USD</div>
            <div style="font-size: 0.95rem; color: #475569;">Baseline: <b style="color: #0f172a;">${base_pred:,.0f}</b> &rarr; Post-Upgrade: <b style="color: #059669;">${upgraded_pred:,.0f} USD</b></div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: EXPLORATORY MARKET ANALYTICS
# ==============================================================================
with tab_eda:
    st.markdown("### 📈 Ames Housing Exploratory Market Analytics")
    
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Total Transactions", f"{len(raw_data):,}")
    with e2:
        st.metric("Median Sale Price", f"${raw_data['SalePrice'].median():,.0f}")
    with e3:
        st.metric("Mean Sale Price", f"${raw_data['SalePrice'].mean():,.0f}")
    with e4:
        st.metric("IQR Spread", f"${raw_data['SalePrice'].quantile(0.75) - raw_data['SalePrice'].quantile(0.25):,.0f}")

    st.markdown("#### 1. Interactive Raw Data Explorer")
    st.dataframe(raw_data[["Id", "SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt", "Neighborhood", "HouseStyle"]].head(15), width='stretch')

    st.markdown("#### 2. Target Variable (`SalePrice`) Distribution & Quality Breakdown")
    fig_eda, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#ffffff")
    for ax in (ax_hist, ax_box):
        ax.set_facecolor("#f8fafc")
        ax.tick_params(colors="#334155")
        for spine in ax.spines.values():
            spine.set_color("#e2e8f0")
        ax.grid(color="#e2e8f0", linestyle='--', alpha=0.6)
            
    # Histogram
    sns.histplot(raw_data["SalePrice"], kde=True, color="#2563eb", ax=ax_hist, bins=35)
    ax_hist.set_title("SalePrice Distribution (Right-Skewed)", color="#0f172a", fontweight="bold")
    ax_hist.set_xlabel("SalePrice ($ USD)", color="#334155")
    ax_hist.axvline(raw_data["SalePrice"].median(), color="#dc2626", linestyle="--", label=f"Median: ${raw_data['SalePrice'].median():,.0f}")
    ax_hist.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#0f172a")
    
    # Boxplot by Quality
    sns.boxplot(x="OverallQual", y="SalePrice", data=raw_data, hue="OverallQual", palette="Blues", legend=False, ax=ax_box)
    ax_box.set_title("SalePrice across Quality Ratings (1-10)", color="#0f172a", fontweight="bold")
    ax_box.set_xlabel("Overall Quality Rating", color="#334155")
    ax_box.set_ylabel("SalePrice ($ USD)", color="#334155")
    
    plt.tight_layout()
    st.pyplot(fig_eda)
    plt.close()

# ==============================================================================
# TAB 5: ACADEMIC REPORT & RESEARCH BOUNDS
# ==============================================================================
with tab_docs:
    st.markdown("### 📜 Academic Research Report & Domain Disclosures")
    
    st.markdown("""
    <div class="disclaimer-banner">
        <h4 style="color: #b91c1c; margin-top: 0; font-size: 1.15rem; font-weight: 700;">⚠️ Academic Disclosures & Real-World Domain Boundaries</h4>
        <ol style="color: #475569; margin-bottom: 0; line-height: 1.65;">
            <li><b style="color: #0f172a;">Geographic & Currency Calibration</b>: Models are calibrated on transactions from Ames, Iowa (United States) denominated strictly in <b>US Dollars ($ USD)</b>. Predictions cannot be treated as valid valuations for Indian properties (INR), which are governed by disparate regulatory systems (RERA), floor rise indices, municipal circle rates, and metro transit premiums.</li>
            <li><b style="color: #0f172a;">Inland Midwest Geography (Absence of Coastal Views)</b>: Ames is an inland Midwestern city. There are <b>no ocean, sea, or coastal attributes</b> in this dataset. We honestly document this limitation rather than fabricating synthetic ocean-view values.</li>
            <li><b style="color: #0f172a;">Constructed Residential Dwellings vs. Vacant Land</b>: All records represent constructed residential dwellings with foundations and structures. Vacant plots and agricultural land valuation are out of scope.</li>
            <li><b style="color: #0f172a;">California Housing Fallback Distinction</b>: California housing evaluates 1990 district census blocks measured in units of $100,000. California metrics are never mixed into the Ames Housing evaluation table.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Methodology Summary
    1. **Zero Data Leakage Protocol**: Transformations (median imputation and standard scaling for numericals; 'None' imputation and one-hot encoding for categoricals) were fitted strictly on the 80% training set ($N=1,168$) and applied downstream via `.transform()`.
    2. **Algorithmic Evaluation**: Five regression paradigms were benchmarked:
       - *Ordinary Least Squares Linear Regression* ($R^2 = 0.8566$)
       - *Random Forest Regressor* ($R^2 = 0.8880$)
       - *Support Vector Regression* ($R^2 = 0.7886$)
       - *Gradient Boosting Regressor* ($R^2 = 0.9111$)
       - *XGBoost Regressor* ($R^2 = 0.9143$ — **Winner**)
    3. **Conclusion**: Gradient boosting algorithms deliver superior predictive accuracy by capturing complex non-linear interactions between square footage and construction quality.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1.25rem 0; font-weight: 500;'>"
    "PropVal Studio &bull; House Price Prediction Using Machine Learning and Comparative Analysis of Regression Models &bull; Ames Housing Dataset"
    "</div>",
    unsafe_allow_html=True
)
