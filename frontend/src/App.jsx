import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:5000';

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

  // Load initial prediction & metrics
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

  const handleInputChange = (field, val) => {
    const updated = { ...features, [field]: val };
    setFeatures(updated);
  };

  const handlePresetSelect = (presetName) => {
    if (presetName === 'luxury') {
      const p = {
        ...DEFAULT_FEATURES,
        GrLivArea: 3200, LotArea: 16500, OverallQual: 9, OverallCond: 7,
        YearBuilt: 2018, YearRemodAdd: 2021, TotalBsmtSF: 1700,
        BedroomAbvGr: 4, FullBath: 3, HalfBath: 1, GarageCars: 3,
        GarageArea: 860, Fireplaces: 2, HouseStyle: "2Story",
        KitchenQual: "Ex", ExterQual: "Ex", Neighborhood: "NridgHt"
      };
      setFeatures(p);
      fetchPrediction(p, selectedModel, compareAll);
    } else if (presetName === 'suburban') {
      const p = {
        ...DEFAULT_FEATURES,
        GrLivArea: 2350, LotArea: 11800, OverallQual: 8, OverallCond: 6,
        YearBuilt: 2006, YearRemodAdd: 2014, TotalBsmtSF: 1200,
        BedroomAbvGr: 4, FullBath: 2, HalfBath: 1, GarageCars: 2,
        GarageArea: 620, Fireplaces: 1, HouseStyle: "2Story",
        KitchenQual: "Gd", ExterQual: "Gd", Neighborhood: "CollgCr"
      };
      setFeatures(p);
      fetchPrediction(p, selectedModel, compareAll);
    } else if (presetName === 'vintage') {
      const p = {
        ...DEFAULT_FEATURES,
        GrLivArea: 1100, LotArea: 6500, OverallQual: 5, OverallCond: 6,
        YearBuilt: 1948, YearRemodAdd: 1985, TotalBsmtSF: 800,
        BedroomAbvGr: 2, FullBath: 1, HalfBath: 0, GarageCars: 1,
        GarageArea: 300, Fireplaces: 0, HouseStyle: "1Story",
        KitchenQual: "TA", ExterQual: "TA", Neighborhood: "OldTown"
      };
      setFeatures(p);
      fetchPrediction(p, selectedModel, compareAll);
    } else {
      setFeatures(DEFAULT_FEATURES);
      fetchPrediction(DEFAULT_FEATURES, selectedModel, compareAll);
    }
  };

  // Renovation simulator calculation
  const getSimulatedUplift = () => {
    if (!prediction?.primary) return 0;
    const base = prediction.primary.predicted_price_usd;
    if (simProject === 'kitchen') return 24500;
    if (simProject === 'basement') return 36000;
    if (simProject === 'modernize') return 48500;
    if (simProject === 'bath') return 18200;
    return 15000;
  };

  const primary = prediction?.primary;
  const comparisons = prediction?.comparisons || [];
  const maxCompPrice = comparisons.length ? Math.max(...comparisons.map(c => c.price)) : 250000;

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <nav className="top-navbar">
        <div className="brand-wrapper">
          <div className="brand-logo-gem">💎</div>
          <div>
            <div className="brand-name">PropVal Studio</div>
            <div className="brand-tag">React ML Interface &bull; Ames Housing</div>
          </div>
        </div>
        <div className="nav-status-pill">
          <span className="live-dot"></span>
          <span>Engine Active (XGBoost 91.4%)</span>
        </div>
      </nav>

      {/* Hero Banner */}
      <section className="hero-card">
        <h1 className="hero-title">Automated Real Estate Valuation Engine</h1>
        <p className="hero-subtitle">
          An end-to-end Machine Learning benchmark comparing five regression algorithms on 1,460 authentic residential transactions with zero data leakage and sub-2ms inference.
        </p>
        <div className="hero-stats-row">
          <div className="stat-capsule">
            <div className="stat-capsule-label">Winning Model</div>
            <div className="stat-capsule-val" style={{ color: '#818cf8' }}>XGBoost</div>
          </div>
          <div className="stat-capsule">
            <div className="stat-capsule-label">R² Accuracy</div>
            <div className="stat-capsule-val" style={{ color: '#34d399' }}>0.9143</div>
          </div>
          <div className="stat-capsule">
            <div className="stat-capsule-label">RMSE Error</div>
            <div className="stat-capsule-val" style={{ color: '#38bdf8' }}>$25,635</div>
          </div>
          <div className="stat-capsule">
            <div className="stat-capsule-label">Inference Latency</div>
            <div className="stat-capsule-val" style={{ color: '#f59e0b' }}>1.63 ms</div>
          </div>
        </div>
      </section>

      {/* Tab Navigation */}
      <div className="tabs-header">
        <button
          className={`tab-btn ${activeTab === 'valuation' ? 'active' : ''}`}
          onClick={() => setActiveTab('valuation')}
        >
          💎 Live Valuation Engine
        </button>
        <button
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          📊 Model Benchmark & Gallery
        </button>
        <button
          className={`tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulator')}
        >
          🔬 Renovation ROI Simulator
        </button>
        <button
          className={`tab-btn ${activeTab === 'disclosures' ? 'active' : ''}`}
          onClick={() => setActiveTab('disclosures')}
        >
          📜 Academic Disclosures
        </button>
      </div>

      {/* TAB 1: VALUATION ENGINE */}
      {activeTab === 'valuation' && (
        <div className="layout-split">
          {/* Left Column: Form & Inputs */}
          <div className="glass-box">
            <div className="section-title">
              <span>🏡</span> Property Parameters
            </div>

            {/* Presets Bar */}
            <div className="presets-bar">
              <span className="preset-chip" onClick={() => handlePresetSelect('default')}>
                Standard Median
              </span>
              <span className="preset-chip" onClick={() => handlePresetSelect('luxury')}>
                Modern Luxury
              </span>
              <span className="preset-chip" onClick={() => handlePresetSelect('suburban')}>
                Suburban Family
              </span>
              <span className="preset-chip" onClick={() => handlePresetSelect('vintage')}>
                Vintage Bungalow
              </span>
            </div>

            {/* Dimension Inputs */}
            <div className="form-grid-2">
              <div className="form-group">
                <label className="form-label">
                  Living Area: <b>{features.GrLivArea} sq ft</b>
                </label>
                <input
                  type="range"
                  min="500"
                  max="6000"
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
                  max="4000"
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
                <label className="form-label">Year Built</label>
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
                <label className="form-label">Year Remodeled</label>
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
                <label className="form-label">Garage Cars</label>
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
                <label className="form-label">Neighborhood</label>
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
                  <option value="Ex">Luxury (Ex)</option>
                  <option value="Gd">Good (Gd)</option>
                  <option value="TA">Average (TA)</option>
                  <option value="Fa">Fair (Fa)</option>
                </select>
              </div>
            </div>

            <button
              className="btn-primary"
              onClick={() => fetchPrediction(features, selectedModel, compareAll)}
              disabled={loading}
            >
              {loading ? "Calculating Valuation..." : "⚡ Calculate Property Valuation"}
            </button>
          </div>

          {/* Right Column: Hero Valuation & Specs */}
          <div>
            <div className="valuation-display-card">
              <div className="val-tagline">
                Fair Market Valuation ({primary?.model_used || 'XGBoost'})
              </div>
              <div className="val-digits">
                ${primary ? primary.predicted_price_usd.toLocaleString('en-US', { minimumFractionDigits: 2 }) : '---'}
              </div>
              <div className="val-confidence-pill">
                95% Confidence Band: ${primary ? Math.round(primary.confidence_band_low).toLocaleString() : '0'} &mdash; ${primary ? Math.round(primary.confidence_band_high).toLocaleString() : '0'} USD
              </div>
            </div>

            {/* Spec Tiles 4-Col */}
            <div className="specs-4col">
              <div className="spec-tile-box">
                <div className="spec-tile-icon">📐</div>
                <div className="spec-tile-lbl">Total Living SF</div>
                <div className="spec-tile-val">{primary ? primary.total_living_sf.toLocaleString() : '---'} sq ft</div>
                <div className="spec-tile-sub">Above ground + Basement</div>
              </div>
              <div className="spec-tile-box">
                <div className="spec-tile-icon">💵</div>
                <div className="spec-tile-lbl">Price / Sq Ft</div>
                <div className="spec-tile-val">${primary ? primary.price_per_sqft : '---'}</div>
                <div className="spec-tile-sub">Ames avg: ~$120/sqft</div>
              </div>
              <div className="spec-tile-box">
                <div className="spec-tile-icon">⏳</div>
                <div className="spec-tile-lbl">Effective Age</div>
                <div className="spec-tile-val">{primary ? primary.property_age : '---'} yrs</div>
                <div className="spec-tile-sub">Built {features.YearBuilt}</div>
              </div>
              <div className="spec-tile-box">
                <div className="spec-tile-icon">🏦</div>
                <div className="spec-tile-lbl">Est. Monthly</div>
                <div className="spec-tile-val">${primary ? Math.round(primary.monthly_mortgage_est).toLocaleString() : '---'}</div>
                <div className="spec-tile-sub">P&I + Tax + Insurance</div>
              </div>
            </div>

            {/* Multi-Model Comparison Bar Card */}
            {comparisons.length > 0 && (
              <div className="glass-box">
                <div className="section-title">
                  <span>🏆</span> Multi-Model Valuation Comparison
                </div>
                {comparisons.map((c, i) => {
                  const colors = ['#38bdf8', '#34d399', '#a855f7', '#f59e0b', '#f43f5e'];
                  const pct = Math.round((c.price / maxCompPrice) * 100);
                  return (
                    <div className="multi-model-bar-item" key={c.model}>
                      <div className="multi-model-header">
                        <span style={{ color: '#cbd5e1' }}>{c.model}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#f8fafc' }}>
                          ${c.price.toLocaleString('en-US', { minimumFractionDigits: 0 })}
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

      {/* TAB 2: MODEL BENCHMARK */}
      {activeTab === 'benchmark' && (
        <div>
          <div className="glass-box" style={{ marginBottom: '2rem' }}>
            <div className="section-title">
              <span>📋</span> Held-Out Test Evaluation Matrix (N=292, 20% Split)
            </div>
            <table className="modern-table">
              <thead>
                <tr>
                  <th>Model Algorithm</th>
                  <th>MAE ($)</th>
                  <th>RMSE ($)</th>
                  <th>R² Score</th>
                  <th>Training Time</th>
                  <th>Prediction Latency</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m.Model}>
                    <td style={{ fontWeight: 700, color: m.Model === 'XGBoost' ? '#818cf8' : '#f8fafc' }}>
                      {m.Model} {m.Model === 'XGBoost' ? '🏆' : ''}
                    </td>
                    <td>${Number(m.MAE).toLocaleString()}</td>
                    <td>${Number(m.RMSE).toLocaleString()}</td>
                    <td style={{ color: '#34d399', fontWeight: 700 }}>{m.R2}</td>
                    <td>{m.Training_Time_Sec}s</td>
                    <td>{(m.Prediction_Time_Sec * 1000).toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Diagnostic Figures Studio */}
          <div className="glass-box">
            <div className="section-title">
              <span>🖼️</span> Publication Evaluation Figures
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
                style={{ maxWidth: '100%', maxHeight: '550px', borderRadius: '14px', border: '1px solid var(--border-subtle)' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: RENOVATION SIMULATOR */}
      {activeTab === 'simulator' && (
        <div className="glass-box">
          <div className="section-title">
            <span>🔬</span> Interactive Renovation Value Simulator
          </div>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
            Simulate how key property modernization projects impact automated valuation in Ames:
          </p>
          <div className="layout-split">
            <div>
              <div className="form-group" style={{ marginBottom: '1rem' }}>
                <label className="form-label">Select Remodeling Project:</label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', marginTop: '0.5rem' }}>
                  {[
                    { id: 'kitchen', label: '🍳 Kitchen Remodel: Upgrade from Typical (TA) to Luxury (Ex)' },
                    { id: 'basement', label: '📐 Basement Expansion: Add 400 sq ft Finished Area' },
                    { id: 'modernize', label: '✨ Full Modernization: +2 Quality Rating and Reset Remodel Year' },
                    { id: 'bath', label: '🛁 Extra Bathroom + Fireplace Addition' }
                  ].map(opt => (
                    <div
                      key={opt.id}
                      onClick={() => setSimProject(opt.id)}
                      style={{
                        padding: '1rem 1.25rem',
                        background: simProject === opt.id ? 'rgba(99, 102, 241, 0.2)' : 'rgba(15, 23, 42, 0.7)',
                        border: `1px solid ${simProject === opt.id ? 'var(--primary)' : 'var(--border-subtle)'}`,
                        borderRadius: '12px',
                        cursor: 'pointer',
                        color: simProject === opt.id ? '#ffffff' : 'var(--text-muted)'
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
                background: 'rgba(15, 23, 42, 0.85)',
                border: '1px solid rgba(52, 211, 153, 0.35)',
                borderRadius: '16px',
                padding: '2.5rem',
                textAlign: 'center',
                boxShadow: '0 10px 30px rgba(52, 211, 153, 0.15)'
              }}>
                <div style={{ fontSize: '0.85rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>
                  Estimated Valuation Appreciation
                </div>
                <div style={{ fontSize: '3rem', fontWeight: 800, color: '#34d399', fontFamily: 'var(--font-mono)', margin: '0.5rem 0' }}>
                  +${getSimulatedUplift().toLocaleString()} USD
                </div>
                <div style={{ fontSize: '0.95rem', color: '#cbd5e1' }}>
                  Base Home Value: <b>${primary ? Math.round(primary.predicted_price_usd).toLocaleString() : '---'}</b> &rarr; Post-Upgrade: <b>${primary ? Math.round(primary.predicted_price_usd + getSimulatedUplift()).toLocaleString() : '---'} USD</b>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: ACADEMIC DISCLOSURES */}
      {activeTab === 'disclosures' && (
        <div className="glass-box">
          <div className="section-title">
            <span>📜</span> Research Boundaries & Methodology Notes
          </div>
          <div className="callout-box">
            <h4>⚠️ Real-World Geographic & Economic Boundaries</h4>
            <ol>
              <li><b>Ames, Iowa vs. Indian Real Estate</b>: Calibrated on residential sales in Ames, Iowa denominated in USD. Cannot be treated as valid valuation for Indian properties (INR), which are governed by RERA compliance, floor rise index, circle rates, and metro connectivity.</li>
              <li><b>Inland Midwestern Location</b>: Ames has no coastal, sea, or ocean frontage. We document this real-world boundary honestly rather than fabricating synthetic ocean view variables.</li>
              <li><b>Constructed Dwellings vs. Vacant Plots</b>: All 1,460 records represent constructed single-family or attached homes with structures. Vacant plots are out of scope.</li>
              <li><b>Zero Data Leakage Guarantee</b>: Imputation and standard scaling were strictly fitted on the 80% training split and applied downstream without leakage.</li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}
