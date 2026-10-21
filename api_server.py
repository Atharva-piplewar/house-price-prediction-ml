"""
REST API Backend Server for PropVal Studio React Interface
Provides machine learning inference and benchmark data over HTTP with full CORS support.
"""

import os
import sys
import json
import time
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.predict import predict_house_price, get_default_property_input, load_model_metadata

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


class PropValAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing REST endpoints for React frontend."""

    def _set_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        """Respond to preflight CORS checks."""
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "healthy",
                "service": "PropVal ML Backend API",
                "champion_model": "XGBoost",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }).encode("utf-8"))

        elif path == "/api/metrics":
            metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                data = df.to_dict(orient="records")
                self._set_headers(200)
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "Metrics summary not found"}).encode("utf-8"))

        elif path == "/api/cv":
            cv_path = os.path.join(RESULTS_DIR, "cv_scores.csv")
            if os.path.exists(cv_path):
                df = pd.read_csv(cv_path)
                data = df.to_dict(orient="records")
                self._set_headers(200)
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "CV scores not found"}).encode("utf-8"))

        elif path == "/api/presets":
            base = get_default_property_input()
            presets = {
                "Standard Median Suburban Home": base,
                "Modern High-End Luxury Residence": {
                    **base,
                    "GrLivArea": 3200, "LotArea": 16500, "OverallQual": 9, "OverallCond": 7,
                    "YearBuilt": 2018, "YearRemodAdd": 2021, "TotalBsmtSF": 1700,
                    "BedroomAbvGr": 4, "FullBath": 3, "HalfBath": 1, "GarageCars": 3,
                    "GarageArea": 860, "Fireplaces": 2, "HouseStyle": "2Story",
                    "KitchenQual": "Ex", "ExterQual": "Ex", "Neighborhood": "NridgHt"
                },
                "Spacious Family Suburban Dwelling": {
                    **base,
                    "GrLivArea": 2350, "LotArea": 11800, "OverallQual": 8, "OverallCond": 6,
                    "YearBuilt": 2006, "YearRemodAdd": 2014, "TotalBsmtSF": 1200,
                    "BedroomAbvGr": 4, "FullBath": 2, "HalfBath": 1, "GarageCars": 2,
                    "GarageArea": 620, "Fireplaces": 1, "HouseStyle": "2Story",
                    "KitchenQual": "Gd", "ExterQual": "Gd", "Neighborhood": "CollgCr"
                },
                "Vintage Character Bungalow": {
                    **base,
                    "GrLivArea": 1100, "LotArea": 6500, "OverallQual": 5, "OverallCond": 6,
                    "YearBuilt": 1948, "YearRemodAdd": 1985, "TotalBsmtSF": 800,
                    "BedroomAbvGr": 2, "FullBath": 1, "HalfBath": 0, "GarageCars": 1,
                    "GarageArea": 300, "Fireplaces": 0, "HouseStyle": "1Story",
                    "KitchenQual": "TA", "ExterQual": "TA", "Neighborhood": "OldTown"
                }
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(presets).encode("utf-8"))

        elif path.startswith("/figures/"):
            filename = os.path.basename(path)
            file_path = os.path.join(FIGURES_DIR, filename)
            if os.path.exists(file_path):
                content_type, _ = mimetypes.guess_type(file_path)
                self._set_headers(200, content_type or "image/png")
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(404)
                self.wfile.write(b"Figure not found")

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"Endpoint {path} not found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body) if body else {}
                prop_features = payload.get("features", get_default_property_input())
                model_name = payload.get("model", None)
                compare_all = payload.get("compare_all", False)

                # Primary prediction
                t0 = time.perf_counter()
                res = predict_house_price(prop_features, model_name=model_name)
                lat_ms = (time.perf_counter() - t0) * 1000

                # Financial calculations
                price = res["predicted_price_usd"]
                gr_area = float(prop_features.get("GrLivArea", 1500))
                bsmt_area = float(prop_features.get("TotalBsmtSF", 800))
                price_per_sqft = price / max(1.0, gr_area)
                
                # Mortgage 30yr 6.8% with 20% down
                loan_amt = price * 0.80
                monthly_rate = 0.068 / 12
                n_payments = 360
                pi = loan_amt * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
                est_tax = (price * 0.012) / 12
                est_ins = 120.0
                monthly_est = pi + est_tax + est_ins

                response_data = {
                    "primary": {
                        **res,
                        "latency_ms": round(lat_ms, 2),
                        "price_per_sqft": round(price_per_sqft, 2),
                        "total_living_sf": gr_area + bsmt_area,
                        "monthly_mortgage_est": round(monthly_est, 2),
                        "confidence_band_low": max(10000.0, price - 25635.0),
                        "confidence_band_high": price + 25635.0
                    }
                }

                # Multi-model comparisons if requested
                if compare_all:
                    all_models = ["Linear Regression", "Random Forest", "Support Vector Regression", "Gradient Boosting", "XGBoost"]
                    comp_list = []
                    for m in all_models:
                        t_start = time.perf_counter()
                        m_res = predict_house_price(prop_features, model_name=m)
                        m_lat = (time.perf_counter() - t_start) * 1000
                        comp_list.append({
                            "model": m,
                            "price": m_res["predicted_price_usd"],
                            "latency_ms": round(m_lat, 2)
                        })
                    response_data["comparisons"] = comp_list

                self._set_headers(200)
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": f"POST endpoint {path} not found"}).encode("utf-8"))


def run_api_server(port=5000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, PropValAPIHandler)
    print(f"[API SERVER] PropVal ML Backend listening on http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_api_server(port)
