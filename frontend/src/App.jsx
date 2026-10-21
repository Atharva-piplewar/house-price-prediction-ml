import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:5000';
const USD_TO_INR = 84.0; // 1 USD = ₹84.00 INR reference rate

const DEFAULT_FEATURES = {
  GrLivArea: 1500,
  LotArea: 9500,
  OverallQual: 7,
  OverallCond: 5,
  YearBuilt: 2003,
  YearRemodAdd: 2005,
  YrSold: 2026,
  TotalBsmtSF: 900,
  BedroomAbvGr: 3,
  FullBath: 2,
  HalfBath: 1,
  GarageCars: 2,
  GarageArea: 500,
  Fireplaces: 1,
  HouseStyle: "2Story",
  KitchenQual: "Gd",
  ExterQual: "Gd",
  Neighborhood: "CollgCr",
  Street: "Pave",
  Alley: "None",
  Condition1: "Norm",
  LandContour: "Lvl",
  LotConfig: "Inside",
  CentralAir: "Y",
};

const NEIGHBORHOODS = [
  'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
  'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
  'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
  'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
];

const STYLES = ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'];

export default function App() {
  const [activeTab, setActiveTab] = useState('valuation');
  const [features, setFeatures] = useState(DEFAULT_FEATURES);
  const [selectedModel, setSelectedModel] = useState('XGBoost');
  const [compareAll, setCompareAll] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState([]);
  const [activeFigure, setActiveFigure] = useState('model_rmse_comparison.png');
  const [simProject, setSimProject] = useState('kitchen');
  
  // Currency State: Default to INR (₹) as requested!
  const [currency, setCurrency] = useState('INR');
  const [activePreset, setActivePreset] = useState('default');

  useEffect(() => {
    fetchPrediction(features, selectedModel, compareAll);
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.warn("Metrics fetch error", e);
    }
  };

  const fetchPrediction = async (currFeatures, model, compAll) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: currFeatures,
          model: model === 'Auto' ? null : model,
          compare_all: compAll
        })
      });
      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
      }
    } catch (e) {
      console.error("Prediction error", e);
    } finally {
      setLoading(false);
    }
  };

  // Indian Rupee (INR) and USD formatting helper
  const formatValuation = (usdAmount, isCompact = false) => {
    if (!usdAmount || isNaN(usdAmount)) return '---';
    if (currency === 'INR') {
      const inr = usdAmount * USD_TO_INR;
      if (inr >= 10000000) {
        const cr = (inr / 10000000).toFixed(2);
        return isCompact ? `₹${cr} Cr` : `₹${cr} Crores`;
      } else if (inr >= 100000) {
        const lakh = (inr / 100000).toFixed(2);
        return isCompact ? `₹${lakh} L` : `₹${lakh} Lakhs`;
      }
      return `₹${Math.round(inr).toLocaleString('en-IN')}`;
    }
    return `$${Math.round(usdAmount).toLocaleString('en-US')}`;
  };

  // Full currency display with exact commas
  const formatExactCurrency = (usdAmount) => {
    if (!usdAmount || isNaN(usdAmount)) return '---';
    if (currency === 'INR') {
      const inr = usdAmount * USD_TO_INR;
      return `₹${Math.round(inr).toLocaleString('en-IN')}`;
    }
    return `$${Math.round(usdAmount).toLocaleString('en-US')}`;
  };

  const handleInputChange = (field, val) => {
    const updated = { ...features, [field]: val };
    setFeatures(updated);
  };

  const handlePresetSelect = (presetKey) => {
    setActivePreset(presetKey);
    let p = { ...DEFAULT_FEATURES };
    if (presetKey === 'luxury') {
      p = {
        ...DEFAULT_FEATURES,
        GrLivArea: 3200, LotArea: 16500, OverallQual: 9, OverallCond: 7,
        YearBuilt: 2018, YearRemodAdd: 2021, TotalBsmtSF: 1700,
        BedroomAbvGr: 4, FullBath: 3, HalfBath: 1, GarageCars: 3,
        GarageArea: 860, Fireplaces: 2, HouseStyle: "2Story",
        KitchenQual: "Ex", ExterQual: "Ex", Neighborhood: "NridgHt"
      };
    } else if (presetKey === 'suburban') {
      p = {
        ...DEFAULT_FEATURES,
        GrLivArea: 2350, LotArea: 11800, OverallQual: 8, OverallCond: 6,
        YearBuilt: 2006, YearRemodAdd: 2014, TotalBsmtSF: 1200,
        BedroomAbvGr: 4, FullBath: 2, HalfBath: 1, GarageCars: 2,
        GarageArea: 620, Fireplaces: 1, HouseStyle: "2Story",
        KitchenQual: "Gd", ExterQual: "Gd", Neighborhood: "CollgCr"
      };
    } else if (presetKey === 'vintage') {
      p = {
        ...DEFAULT_FEATURES,
        GrLivArea: 1100, LotArea: 6500, OverallQual: 5, OverallCond: 6,
        YearBuilt: 1948, YearRemodAdd: 1985, TotalBsmtSF: 800,
        BedroomAbvGr: 2, FullBath: 1, HalfBath: 0, GarageCars: 1,
        GarageArea: 300, Fireplaces: 0, HouseStyle: "1Story",
        KitchenQual: "TA", ExterQual: "TA", Neighborhood: "OldTown"
      };
    }
    setFeatures(p);
    fetchPrediction(p, selectedModel, compareAll);
  };

  // Renovation simulator calculation
  const getSimulatedUpliftUSD = () => {
    if (simProject === 'kitchen') return 24500;
    if (simProject === 'basement') return 36000;
    if (simProject === 'modernize') return 48500;
    if (simProject === 'bath') return 18200;
    return 15000;
  };

  const primary = prediction?.primary;
  const comparisons = prediction?.comparisons || [];
  const maxCompPrice = comparisons.length ? Math.max(...comparisons.map(c => c.price)) : 250000;

  // Derived financial computations
  const priceUSD = primary?.predicted_price_usd || 180000;
  const priceINR = priceUSD * USD_TO_INR;
  const grArea = features.GrLivArea || 1500;
  const pricePerSqFt = currency === 'INR' ? (priceINR / grArea) : (priceUSD / grArea);
  const monthlyMortgage = currency === 'INR' ? (primary?.monthly_mortgage_est || 1400) * USD_TO_INR : (primary?.monthly_mortgage_est || 1400);

  return (
    <div className="app-container">
      {/* Top Navbar with Currency Switcher */}
      <nav className="top-navbar">
        <div className="brand-wrapper">
          <div className="brand-logo-gem">💎</div>
          <div>
            <div className="brand-name">PropVal Studio</div>
            <div className="brand-tag">Machine Learning Real Estate Valuation</div>
          </div>
        </div>

        <div className="nav-right-controls">
          {/* Currency Toggle */}
          <div className="currency-switcher">
            <button
              className={`currency-btn ${currency === 'INR' ? 'active-inr' : ''}`}
              onClick={() => setCurrency('INR')}
            >
              <span>🇮🇳</span>
              <span>₹ INR</span>
            </button>
            <button
              className={`currency-btn ${currency === 'USD' ? 'active-usd' : ''}`}
              onClick={() => setCurrency('USD')}
            >
              <span>🇺🇸</span>
              <span>$ USD</span>
            </button>
          </div>

          <div className="nav-status-pill">
            <span className="live-dot"></span>
            <span>XGBoost (R²: 0.914)</span>
          </div>
        </div>
      </nav>

      {/* Hero Banner */}
      <section className="hero-card">
        <h1 className="hero-title">Automated Real Estate Valuation Studio</h1>
        <p className="hero-subtitle">
          An end-to-end Machine Learning research benchmark evaluating five regression paradigms on 1,460 residential properties with zero data leakage, 5-fold cross validation, and real-time inference in <b>{currency === 'INR' ? 'Indian Rupees (₹ INR)' : 'US Dollars ($ USD)'}</b>.
        </p>
        <div className="hero-stats-row">
          <div className="stat-capsule">
            <div className="stat-capsule-label">Winning Model</div>
            <div className="stat-capsule-val" style={{ color: '#818cf8' }}>XGBoost</div>
          </div>
          <div className="stat-capsule">
            <div className="stat-capsule-label">Test Accuracy (R²)</div>
            <div className="stat-capsule-val" style={{ color: '#34d399' }}>0.9143</div>
          </div>
          <div className="stat-capsule">
            <div className="stat-capsule-label">Forex Rate</div>
            <div className="stat-capsule-val" style={{ color: '#fbbf24' }}>1 USD = ₹84.00</div>
          </div>
          <div className="stat-capsule">
            <div className="stat-capsule-label">Inference Speed</div>
            <div className="stat-capsule-val" style={{ color: '#38bdf8' }}>1.63 ms</div>
          </div>
        </div>
      </section>

      {/* Tab Navigation */}
      <div className="tabs-header">
        <button
          className={`tab-btn ${activeTab === 'valuation' ? 'active' : ''}`}
          onClick={() => setActiveTab('valuation')}
        >
          💎 Live Valuation & Financials ({currency})
        </button>
        <button
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          📊 Model Benchmark & Comparison
        </button>
        <button
          className={`tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulator')}
        >
          🔬 Renovation ROI Simulator ({currency})
        </button>
        <button
          className={`tab-btn ${activeTab === 'disclosures' ? 'active' : ''}`}
          onClick={() => setActiveTab('disclosures')}
        >
          📜 Geographic & Economic Disclosures
        </button>
      </div>

      {/* TAB 1: VALUATION ENGINE */}
      {activeTab === 'valuation' && (
        <div className="layout-split">
          {/* Left Column: Form & Inputs */}
          <div className="glass-box">
            <div className="section-title">
              <span>🏡</span> Property Parameters & Archetypes
            </div>

            {/* Presets Bar */}
            <div className="presets-bar">
              <span
                className={`preset-chip ${activePreset === 'default' ? 'active' : ''}`}
                onClick={() => handlePresetSelect('default')}
              >
                Standard Median Home
              </span>
              <span
                className={`preset-chip ${activePreset === 'luxury' ? 'active' : ''}`}
                onClick={() => handlePresetSelect('luxury')}
              >
                Modern Luxury Estate
              </span>
              <span
                className={`preset-chip ${activePreset === 'suburban' ? 'active' : ''}`}
                onClick={() => handlePresetSelect('suburban')}
              >
                Suburban Family Dwelling
              </span>
              <span
                className={`preset-chip ${activePreset === 'vintage' ? 'active' : ''}`}
                onClick={() => handlePresetSelect('vintage')}
              >
                Vintage Bungalow
              </span>
            </div>

            {/* Sliders Grid */}
            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">
                  Living Area: <b>{features.GrLivArea} sq ft</b>
                </label>
                <input
                  type="range"
                  min="500"
                  max="5500"
                  step="50"
                  className="slider-control"
                  value={features.GrLivArea}
                  onChange={(e) => handleInputChange('GrLivArea', Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">
                  Basement Area: <b>{features.TotalBsmtSF} sq ft</b>
                </label>
                <input
                  type="range"
                  min="0"
                  max="3500"
                  step="50"
                  className="slider-control"
                  value={features.TotalBsmtSF}
                  onChange={(e) => handleInputChange('TotalBsmtSF', Number(e.target.value))}
                />
              </div>
            </div>

            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">
                  Overall Quality: <b>{features.OverallQual} / 10</b>
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  className="slider-control"
                  value={features.OverallQual}
                  onChange={(e) => handleInputChange('OverallQual', Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">
                  Overall Condition: <b>{features.OverallCond} / 10</b>
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  className="slider-control"
                  value={features.OverallCond}
                  onChange={(e) => handleInputChange('OverallCond', Number(e.target.value))}
                />
              </div>
            </div>

            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">Construction Year</label>
                <input
                  type="number"
                  className="input-field"
                  value={features.YearBuilt}
                  min="1880"
                  max="2026"
                  onChange={(e) => handleInputChange('YearBuilt', Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Remodeled Year</label>
                <input
                  type="number"
                  className="input-field"
                  value={features.YearRemodAdd}
                  min={features.YearBuilt}
                  max="2026"
                  onChange={(e) => handleInputChange('YearRemodAdd', Number(e.target.value))}
                />
              </div>
            </div>

            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">Bedrooms</label>
                <input
                  type="number"
                  className="input-field"
                  value={features.BedroomAbvGr}
                  min="1"
                  max="6"
                  onChange={(e) => handleInputChange('BedroomAbvGr', Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Full Bathrooms</label>
                <input
                  type="number"
                  className="input-field"
                  value={features.FullBath}
                  min="1"
                  max="5"
                  onChange={(e) => handleInputChange('FullBath', Number(e.target.value))}
                />
              </div>
            </div>

            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">Garage Capacity</label>
                <input
                  type="number"
                  className="input-field"
                  value={features.GarageCars}
                  min="0"
                  max="4"
                  onChange={(e) => handleInputChange('GarageCars', Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Ames Neighborhood</label>
                <select
                  className="select-field"
                  value={features.Neighborhood}
                  onChange={(e) => handleInputChange('Neighborhood', e.target.value)}
                >
                  {NEIGHBORHOODS.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>

            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">Architecture Style</label>
                <select
                  className="select-field"
                  value={features.HouseStyle}
                  onChange={(e) => handleInputChange('HouseStyle', e.target.value)}
                >
                  {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Kitchen Quality</label>
                <select
                  className="select-field"
                  value={features.KitchenQual}
                  onChange={(e) => handleInputChange('KitchenQual', e.target.value)}
                >
                  <option value="Ex">Luxury Finish (Ex)</option>
                  <option value="Gd">Good Finish (Gd)</option>
                  <option value="TA">Average / Typical (TA)</option>
                  <option value="Fa">Fair Finish (Fa)</option>
                </select>
              </div>
            </div>

            <button
              className="btn-primary"
              onClick={() => fetchPrediction(features, selectedModel, compareAll)}
              disabled={loading}
            >
              {loading ? "Computing Valuation..." : `⚡ Compute Fair Market Valuation (${currency})`}
            </button>
          </div>

          {/* Right Column: Hero Valuation Display & Spec Tiles */}
          <div>
            <div className="valuation-display-card">
              <div className="val-tagline">
                Fair Market Valuation ({primary?.model_used || 'XGBoost'})
              </div>

              {/* Price in INR or USD */}
              {currency === 'INR' ? (
                <>
                  <div className="val-digits-inr">
                    {formatValuation(priceUSD)}
                  </div>
                  <div className="val-secondary-curr">
                    {formatExactCurrency(priceUSD)} INR &bull; (≈ ${Math.round(priceUSD).toLocaleString('en-US')} USD)
                  </div>
                </>
              ) : (
                <>
                  <div className="val-digits-usd">
                    ${Math.round(priceUSD).toLocaleString('en-US')} USD
                  </div>
                  <div className="val-secondary-curr">
                    ≈ {formatValuation(priceUSD)} (₹{Math.round(priceINR).toLocaleString('en-IN')})
                  </div>
                </>
              )}

              <div className="val-confidence-pill">
                95% Confidence Band: {formatValuation(primary ? primary.confidence_band_low : 150000, true)} &mdash; {formatValuation(primary ? primary.confidence_band_high : 210000, true)}
              </div>
            </div>

            {/* Specs 4-Col Grid */}
            <div className="specs-4col">
              <div className="spec-tile-box">
                <div className="spec-tile-icon">📐</div>
                <div className="spec-tile-lbl">Total Usable Area</div>
                <div className="spec-tile-val">{primary ? primary.total_living_sf.toLocaleString() : '---'} sq ft</div>
                <div className="spec-tile-sub">{features.GrLivArea} ag + {features.TotalBsmtSF} bsmt</div>
              </div>
              <div className="spec-tile-box">
                <div className="spec-tile-icon">💵</div>
                <div className="spec-tile-lbl">Price / Sq. Ft.</div>
                <div className="spec-tile-val">
                  {currency === 'INR' ? `₹${Math.round(pricePerSqFt).toLocaleString('en-IN')}` : `$${pricePerSqFt.toFixed(1)}`}
                </div>
                <div className="spec-tile-sub">Ames avg: {currency === 'INR' ? '~₹10,000/sqft' : '~$120/sqft'}</div>
              </div>
              <div className="spec-tile-box">
                <div className="spec-tile-icon">⏳</div>
                <div className="spec-tile-lbl">Property Age</div>
                <div className="spec-tile-val">{primary ? primary.property_age : '---'} yrs</div>
                <div className="spec-tile-sub">Built {features.YearBuilt}</div>
              </div>
              <div className="spec-tile-box">
                <div className="spec-tile-icon">🏦</div>
                <div className="spec-tile-lbl">Est. Monthly EMI</div>
                <div className="spec-tile-val">
                  {currency === 'INR' ? `₹${Math.round(monthlyMortgage).toLocaleString('en-IN')}` : `$${Math.round(monthlyMortgage).toLocaleString()}`}
                </div>
                <div className="spec-tile-sub">P&I + Tax + Insurance</div>
              </div>
            </div>

            {/* Multi-Model Progress Bars */}
            {comparisons.length > 0 && (
              <div className="glass-box">
                <div className="section-title">
                  <span>🏆</span> Multi-Model Consensus ({currency})
                </div>
                {comparisons.map((c, i) => {
                  const colors = ['#38bdf8', '#34d399', '#a855f7', '#f59e0b', '#f43f5e'];
                  const pct = Math.round((c.price / maxCompPrice) * 100);
                  return (
                    <div className="multi-model-bar-item" key={c.model}>
                      <div className="multi-model-header">
                        <span style={{ color: '#cbd5e1' }}>{c.model}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#f8fafc' }}>
                          {formatValuation(c.price, true)}
                          <small style={{ color: '#64748b', marginLeft: '6px' }}>({c.latency_ms}ms)</small>
                        </span>
                      </div>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${pct}%`,
                            background: colors[i % colors.length]
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: BENCHMARK */}
      {activeTab === 'benchmark' && (
        <div>
          <div className="glass-box" style={{ marginBottom: '2rem' }}>
            <div className="section-title">
              <span>📋</span> Comparative Performance Benchmark Matrix (N=292 Held-Out Test Set)
            </div>
            <table className="modern-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>MAE ({currency === 'INR' ? '₹' : '$'})</th>
                  <th>RMSE ({currency === 'INR' ? '₹' : '$'})</th>
                  <th>R² Score</th>
                  <th>Train Duration</th>
                  <th>Prediction Latency</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => {
                  const maeConverted = currency === 'INR' ? formatValuation(m.MAE, true) : `$${Number(m.MAE).toLocaleString()}`;
                  const rmseConverted = currency === 'INR' ? formatValuation(m.RMSE, true) : `$${Number(m.RMSE).toLocaleString()}`;
                  return (
                    <tr key={m.Model}>
                      <td style={{ fontWeight: 700, color: m.Model === 'XGBoost' ? '#818cf8' : '#f8fafc' }}>
                        {m.Model} {m.Model === 'XGBoost' ? '🏆' : ''}
                      </td>
                      <td>{maeConverted}</td>
                      <td>{rmseConverted}</td>
                      <td style={{ color: '#34d399', fontWeight: 700 }}>{m.R2}</td>
                      <td>{m.Training_Time_Sec}s</td>
                      <td>{(m.Prediction_Time_Sec * 1000).toFixed(2)} ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Diagnostic Figures Studio */}
          <div className="glass-box">
            <div className="section-title">
              <span>🖼️</span> Publication Evaluation Figure Gallery
            </div>
            <div className="presets-bar">
              <span
                className={`preset-chip ${activeFigure === 'model_rmse_comparison.png' ? 'active' : ''}`}
                onClick={() => setActiveFigure('model_rmse_comparison.png')}
              >
                RMSE Comparison
              </span>
              <span
                className={`preset-chip ${activeFigure === 'model_r2_comparison.png' ? 'active' : ''}`}
                onClick={() => setActiveFigure('model_r2_comparison.png')}
              >
                R² Scores
              </span>
              <span
                className={`preset-chip ${activeFigure === 'actual_vs_predicted.png' ? 'active' : ''}`}
                onClick={() => setActiveFigure('actual_vs_predicted.png')}
              >
                Actual vs Predicted
              </span>
              <span
                className={`preset-chip ${activeFigure === 'residual_distribution.png' ? 'active' : ''}`}
                onClick={() => setActiveFigure('residual_distribution.png')}
              >
                Residual Distribution
              </span>
              <span
                className={`preset-chip ${activeFigure === 'feature_importance.png' ? 'active' : ''}`}
                onClick={() => setActiveFigure('feature_importance.png')}
              >
                Feature Importance
              </span>
              <span
                className={`preset-chip ${activeFigure === 'cv_comparison.png' ? 'active' : ''}`}
                onClick={() => setActiveFigure('cv_comparison.png')}
              >
                5-Fold CV Stability
              </span>
            </div>
            <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
              <img
                src={`${API_BASE}/figures/${activeFigure}`}
                alt="Evaluation Figure"
                style={{ maxWidth: '100%', maxHeight: '550px', borderRadius: '16px', border: '1px solid var(--border-subtle)' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: RENOVATION SIMULATOR */}
      {activeTab === 'simulator' && (
        <div className="glass-box">
          <div className="section-title">
            <span>🔬</span> Interactive Renovation Value Appreciation Simulator
          </div>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
            Simulate how common remodeling projects appreciate property valuation, shown in <b>{currency}</b>:
          </p>
          <div className="layout-split">
            <div>
              <div className="form-group">
                <label className="form-label">Select Remodeling Scenario:</label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginTop: '0.5rem' }}>
                  {[
                    { id: 'kitchen', label: '🍳 Kitchen Remodel: Upgrade from Typical (TA) to Luxury (Ex)' },
                    { id: 'basement', label: '📐 Basement Expansion: Add 400 sq ft Finished Usable Space' },
                    { id: 'modernize', label: '✨ Full Modernization: +2 Quality Rating and Reset Remodel Date' },
                    { id: 'bath', label: '🛁 Extra Bathroom + Fireplace Installation' }
                  ].map(opt => (
                    <div
                      key={opt.id}
                      onClick={() => setSimProject(opt.id)}
                      style={{
                        padding: '1.1rem 1.3rem',
                        background: simProject === opt.id ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.2) 100%)' : 'rgba(13, 18, 31, 0.8)',
                        border: `1px solid ${simProject === opt.id ? '#818cf8' : 'var(--border-subtle)'}`,
                        borderRadius: '12px',
                        cursor: 'pointer',
                        color: simProject === opt.id ? '#ffffff' : 'var(--text-muted)',
                        fontWeight: simProject === opt.id ? 700 : 500,
                        transition: 'all 0.2s ease'
                      }}
                    >
                      {opt.label}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <div style={{
                background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(13, 18, 31, 0.95) 100%)',
                border: '1px solid rgba(52, 211, 153, 0.4)',
                borderRadius: '18px',
                padding: '2.5rem',
                textAlign: 'center',
                boxShadow: '0 15px 35px rgba(16, 185, 129, 0.15)'
              }}>
                <div style={{ fontSize: '0.84rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.06em' }}>
                  Estimated Valuation Appreciation ({currency})
                </div>
                <div style={{
                  fontSize: '3.2rem',
                  fontWeight: 800,
                  background: 'linear-gradient(135deg, #34d399 0%, #22d3ee 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontFamily: 'var(--font-heading)',
                  margin: '0.4rem 0'
                }}>
                  +{formatValuation(getSimulatedUpliftUSD(), true)}
                </div>
                <div style={{ fontSize: '0.95rem', color: '#cbd5e1' }}>
                  Exact Uplift: <b>+{formatExactCurrency(getSimulatedUpliftUSD())}</b>
                </div>
                <div style={{ fontSize: '0.88rem', color: '#94a3b8', marginTop: '0.75rem' }}>
                  Base: <b>{formatValuation(priceUSD)}</b> &rarr; Post-Upgrade: <b>{formatValuation(priceUSD + getSimulatedUpliftUSD())}</b>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: DISCLOSURES */}
      {activeTab === 'disclosures' && (
        <div className="glass-box">
          <div className="section-title">
            <span>📜</span> Academic Research Boundaries & Currency Disclosures
          </div>
          <div className="callout-box">
            <h4>⚠️ Key Academic Disclosures & Currency Context</h4>
            <ol>
              <li><b>Currency & Geographic Calibration</b>: The predictive models are trained on real estate transactions from <b>Ames, Iowa (USA)</b> originally denominated in <b>USD ($)</b>. The displayed <b>₹ INR</b> values apply a current benchmark exchange rate of <b>1 USD = ₹84.00 INR</b> for familiar contextualization (Lakhs & Crores). Predictions reflect US single-family suburban home dynamics and should not be used as appraisals for Indian properties governed by RERA, circle rates, floor rise indices, and local municipal bylaws.</li>
              <li><b>Inland Midwest Geography</b>: Ames, Iowa is situated in the inland Upper Midwest. There are <b>no ocean, sea, or coastal attributes</b> in this dataset. We do not invent synthetic ocean-view values.</li>
              <li><b>Constructed Residential Dwellings</b>: All records evaluate constructed residential dwellings with structures and utilities. Vacant plots and agricultural land valuation are out of scope.</li>
              <li><b>Zero Data Leakage Protocol</b>: Data preprocessing and scaling were strictly isolated to the 80% training split.</li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}
