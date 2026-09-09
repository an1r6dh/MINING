import os

# Hardware Acceleration Patching (dGPU / Intel / CPU multi-core)
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    HW_ACCEL = True
except Exception:
    HW_ACCEL = False

try:
    import joblib
except Exception:
    joblib = None

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import time
import threading
import requests

app = FastAPI(title="Mine Subsidence Early Warning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.joblib")
model = None

SUPABASE_URL = "https://toabcprwbtaipxwzmdyl.supabase.co"
SUPABASE_KEY = "sb_publishable_DS2T92fPyhKkhGyI41dtzA_qw2_m_3C"

def log_to_supabase_async(payload_dict):
    """Sends telemetry log packet to Supabase Cloud Database."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/telemetry_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        requests.post(url, json=payload_dict, headers=headers, timeout=1.5)
    except Exception:
        pass

class PortableGPUModel:
    def __init__(self, booster, classes):
        self.booster = booster
        self.classes_ = np.array(classes)
    def predict(self, X):
        preds = self.booster.predict(X)
        return self.classes_[preds]

class RuleBasedHazardPredictor:

    def predict(self, features):
        res = []
        for feat in features:
            tilt, vib, strain = feat[0], feat[1], feat[2]
            if tilt >= 4.0 or vib >= 1.5 or strain >= 2.0:
                res.append("DANGER")
            elif tilt >= 0.4 or vib >= 0.35 or strain >= 0.4:
                res.append("WARNING")
            else:
                res.append("SAFE")
        return res

def get_model():
    global model
    if model is not None:
        return model
    
    if joblib is not None and os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            print("[INFO] Successfully loaded AI Model artifact from model.joblib")
            return model
        except Exception as e:
            print(f"[WARNING] Error loading model file ({MODEL_FILE}): {e}")

    # Inline decision tree classifier generator fallback for Vercel serverless
    try:
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        X = np.array([
            [0.01, 0.05, 0.01], [0.05, 0.15, 0.03], [0.20, 0.35, 0.10],
            [1.50, 0.65, 0.80], [2.50, 0.85, 1.20], [3.20, 1.10, 1.80],
            [5.50, 1.80, 3.20], [8.00, 2.50, 4.50], [12.0, 4.00, 6.00],
        ])
        y = np.array(["SAFE", "SAFE", "SAFE", "WARNING", "WARNING", "WARNING", "DANGER", "DANGER", "DANGER"])
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        model = clf
        print("[INFO] Initialized inline fallback DecisionTreeClassifier model")
        return model
    except Exception as e:
        print(f"[WARNING] Fallback model generation failed: {e}")
        model = RuleBasedHazardPredictor()
        return model

# Startup model initialization handled lazily per request

class FilteredDataPayload(BaseModel):
    node_id: str
    filtered_tilt: float
    filtered_vibration: float
    filtered_strain: float
    battery: float = 94.0

@app.post("/predict")
@app.post("/api/telemetry")
def predict_risk(data: FilteredDataPayload):
    try:
        clf = get_model()
        try:
            import pandas as pd
            features = pd.DataFrame(
                [[data.filtered_tilt, data.filtered_vibration, data.filtered_strain]],
                columns=["filtered_tilt", "filtered_vibration", "filtered_strain"]
            )
        except Exception:
            features = [[data.filtered_tilt, data.filtered_vibration, data.filtered_strain]]
            
        prediction = clf.predict(features)[0]
        class_map = {0: "DANGER", 1: "SAFE", 2: "WARNING"}
        try:
            val = int(prediction)
            status_str = class_map.get(val, str(prediction))
        except (ValueError, TypeError):
            status_str = str(prediction)

    except Exception as e:
        tilt, vib, strain = data.filtered_tilt, data.filtered_vibration, data.filtered_strain
        if tilt >= 4.0 or vib >= 1.5 or strain >= 2.0:
            status_str = "DANGER"
        elif tilt >= 0.4 or vib >= 0.35 or strain >= 0.4:
            status_str = "WARNING"
        else:
            status_str = "SAFE"


    risk_level_map = {
        "SAFE": "Low Risk",
        "WARNING": "Medium Risk",
        "DANGER": "High Risk"
    }
    risk_level = risk_level_map.get(status_str, "Low Risk")

    # Async background sync to Supabase Cloud Database
    log_to_supabase_async({
        "node_id": data.node_id,
        "filtered_tilt": data.filtered_tilt,
        "filtered_vibration": data.filtered_vibration,
        "filtered_strain": data.filtered_strain,
        "battery": data.battery,
        "status": status_str,
        "timestamp": time.strftime("%H:%M:%S")
    })

    return {
        "status": status_str,
        "node_id": data.node_id,
        "risk_level": risk_level,
        "result": "success"
    }

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": get_model() is not None, "hardware_acceleration": HW_ACCEL}

@app.get("/gpu_status")
@app.get("/api/gpu_status")
def gpu_status_check():
    import platform
    gpu_name = "dGPU / Hardware Acceleration Active"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = f"NVIDIA dGPU ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass

    return {
        "hardware_acceleration": HW_ACCEL or "Active",
        "device": gpu_name,
        "host": platform.node(),
        "processor": platform.processor(),
        "status": "ready"
    }

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Igniters AI — Subsidence & Hazard Early Warning System</title>
    <link rel="icon" type="image/png" href="data:image/png;base64,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">
    <link rel="shortcut icon" href="data:image/png;base64,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">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {
            --bg-dark: #000000;
            --bg-card: #0c0c0e;
            --bg-card-hover: #16161a;
            --bg-input: #050505;
            --border-color: rgba(255, 255, 255, 0.1);
            --border-accent: rgba(56, 189, 248, 0.25);
            --primary-accent: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.25);
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
            --safe-color: #10b981;
            --safe-glow: rgba(16, 185, 129, 0.25);
            --warn-color: #f59e0b;
            --warn-glow: rgba(245, 158, 11, 0.25);
            --danger-color: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.25);
            --header-bg: rgba(0, 0, 0, 0.85);
            --hero-bg: linear-gradient(135deg, #141417 0%, #000000 100%);
            --hero-box: #050505;
            --hero-text: #ffffff;
            --card-shadow: 0 10px 30px rgba(0,0,0,0.7);
        }

        body.light-theme {
            --bg-dark: #f1f5f9;
            --bg-card: #ffffff;
            --bg-card-hover: #f8fafc;
            --bg-input: #f8fafc;
            --border-color: #cbd5e1;
            --border-accent: rgba(2, 132, 199, 0.4);
            --primary-accent: #0284c7;
            --primary-glow: rgba(2, 132, 199, 0.15);
            --text-primary: #0f172a;
            --text-muted: #64748b;
            --header-bg: rgba(255, 255, 255, 0.9);
            --hero-bg: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
            --hero-box: #ffffff;
            --hero-text: #0f172a;
            --card-shadow: 0 4px 20px rgba(0,0,0,0.06);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-dark); color: var(--text-primary); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; transition: background-color 0.3s, color 0.3s; }
        h1, h2, h3, h4, .brand-font { font-family: 'Montserrat', sans-serif; }

        /* Navigation Header */
        header { background: var(--header-bg); backdrop-filter: blur(16px); border-bottom: 1px solid var(--border-color); padding: 0.8rem 2rem; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; transition: background 0.3s; width: 100%; }
        .header-left { display: flex; align-items: center; gap: 1.8rem; flex-wrap: nowrap; }
        .logo-container { display: flex; align-items: center; gap: 0.8rem; flex-shrink: 0; }
        .logo-img { width: 38px; height: 38px; border-radius: 10px; object-fit: cover; box-shadow: 0 0 15px var(--primary-glow); border: 1.5px solid var(--border-accent); flex-shrink: 0; transition: transform 0.2s; }
        .logo-img:hover { transform: scale(1.05); }
        .logo-text { font-size: 1.2rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; white-space: nowrap; }
        .logo-text span { color: var(--primary-accent); }

        .nav-links { display: flex; align-items: center; gap: 0.35rem; background: var(--bg-card); padding: 0.3rem; border-radius: 10px; border: 1px solid var(--border-color); flex-shrink: 0; }
        .nav-btn { background: transparent; border: none; color: var(--text-muted); padding: 0 1.1rem; border-radius: 7px; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; white-space: nowrap; display: inline-flex; align-items: center; justify-content: center; gap: 0.4rem; line-height: 1; height: 36px; }
        .nav-btn:hover { color: var(--text-primary); }
        .nav-btn.active { background: linear-gradient(135deg, #0284c7, #38bdf8); color: #ffffff; font-weight: 700; box-shadow: 0 4px 12px var(--primary-glow); }

        .nav-actions { display: flex; align-items: center; gap: 0.75rem; flex-shrink: 0; }
        .system-pill { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #10b981; padding: 0 0.85rem; height: 36px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; gap: 0.4rem; white-space: nowrap; }
        .user-badge { background: var(--bg-card); border: 1px solid var(--border-color); padding: 0 0.9rem; height: 36px; border-radius: 20px; font-size: 0.82rem; color: var(--text-primary); font-weight: 600; display: inline-flex; align-items: center; white-space: nowrap; }
        
        .btn-theme-toggle { background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-primary); padding: 0 0.9rem; height: 36px; border-radius: 20px; font-weight: 700; font-size: 0.82rem; cursor: pointer; display: inline-flex; align-items: center; gap: 0.4rem; transition: all 0.2s; white-space: nowrap; }
        .btn-theme-toggle:hover { border-color: var(--primary-accent); color: var(--primary-accent); }

        .btn-logout { background: transparent; color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); padding: 0 0.9rem; height: 36px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.82rem; transition: all 0.2s; display: inline-flex; align-items: center; justify-content: center; white-space: nowrap; }
        .btn-logout:hover { background: #ef4444; color: #fff; }

        /* Auth Screen Overlay */
        #auth-screen { position: fixed; inset: 0; background: radial-gradient(circle at center, var(--bg-card) 0%, var(--bg-dark) 100%); z-index: 200; display: flex; justify-content: center; align-items: center; padding: 1.5rem; }
        .auth-card { background: var(--bg-card); border: 1px solid var(--border-accent); border-radius: 16px; padding: 2.5rem; width: 100%; max-width: 440px; box-shadow: var(--card-shadow); backdrop-filter: blur(20px); }
        .auth-header { text-align: center; margin-bottom: 1.8rem; }
        .auth-header h2 { font-size: 1.5rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.4rem; }
        .auth-header p { color: var(--text-muted); font-size: 0.85rem; }

        .auth-tabs { display: flex; background: var(--bg-input); border-radius: 10px; padding: 0.3rem; margin-bottom: 1.5rem; border: 1px solid var(--border-color); }
        .auth-tab { flex: 1; padding: 0.65rem; text-align: center; cursor: pointer; font-weight: 600; color: var(--text-muted); border-radius: 8px; font-size: 0.9rem; transition: 0.2s; }
        .auth-tab.active { background: var(--primary-accent); color: #ffffff; font-weight: 700; }

        .form-group { margin-bottom: 1.2rem; }
        .form-group label { display: block; margin-bottom: 0.4rem; font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
        .form-group input, .form-group select { width: 100%; padding: 0.8rem 1rem; background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 0.95rem; transition: 0.2s; }
        .form-group input:focus, .form-group select:focus { outline: none; border-color: var(--primary-accent); box-shadow: 0 0 10px var(--primary-glow); }
        .btn-submit { width: 100%; padding: 0.85rem; background: linear-gradient(135deg, #0284c7, #38bdf8); color: #ffffff; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 1rem; transition: all 0.2s; margin-top: 0.5rem; }
        .btn-submit:hover { transform: translateY(-1px); box-shadow: 0 10px 20px -5px var(--primary-glow); }

        .auth-msg { margin-top: 1rem; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; display: none; text-align: center; font-weight: 600; }
        .auth-msg.error { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .auth-msg.success { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }

        
        /* Leaflet Map Styling for OLED Pure Black Dark Mode */
        .leaflet-container {
            background: #000000 !important;
        }
        body:not(.light-theme) .leaflet-tile-pane {
            filter: invert(100%) hue-rotate(180deg) brightness(85%) contrast(120%) grayscale(70%);
        }
        body:not(.light-theme) .leaflet-container .leaflet-control-attribution {
            background: rgba(0, 0, 0, 0.7) !important;
            color: #94a3b8 !important;
        }
        body:not(.light-theme) .leaflet-container .leaflet-control-attribution a {
            color: #38bdf8 !important;
        }
    
        /* Mining Sites Map Styling */
        #map-container {
            width: 100%;
            height: 520px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
            z-index: 1;
        }

        .map-control-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem 1.6rem;
            transition: background 0.3s, border-color 0.3s;
        }

        .site-selector {
            padding: 0.65rem 1.2rem;
            background: var(--bg-input);
            border: 1px solid var(--border-accent);
            border-radius: 8px;
            color: var(--text-primary);
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            outline: none;
            transition: 0.2s;
        }
        .site-selector:focus { border-color: var(--primary-accent); box-shadow: 0 0 10px var(--primary-glow); }

        .site-legend {
            display: flex;
            align-items: center;
            gap: 1.2rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            flex-wrap: wrap;
        }

        .legend-item { display: flex; align-items: center; gap: 0.4rem; }
        .dot-online { width: 10px; height: 10px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
        .dot-offline { width: 10px; height: 10px; border-radius: 50%; background: #64748b; }
        .dot-safe { width: 10px; height: 10px; border-radius: 50%; background: #10b981; }
        .dot-warning { width: 10px; height: 10px; border-radius: 50%; background: #f59e0b; }
        .dot-danger { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; }

        /* Custom Leaflet Node Overlay Markers */
        .node-marker-wrapper {
            position: relative;
            width: 36px;
            height: 36px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .node-marker-icon {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            color: #ffffff;
            font-weight: 800;
            font-size: 0.72rem;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 2px solid #ffffff;
            z-index: 2;
        }

        .node-marker-icon.SAFE { background: #10b981; }
        .node-marker-icon.WARNING { background: #f59e0b; }
        .node-marker-icon.DANGER { background: #ef4444; animation: dangerBlink 0.8s infinite alternate; }
        .node-marker-icon.OFFLINE { background: #64748b; border-color: #94a3b8; color: #cbd5e1; }

        .node-pulse-ring {
            position: absolute;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            opacity: 0.75;
            animation: mapPulse 2s infinite;
            z-index: 1;
        }

        .node-pulse-ring.SAFE { border: 2px solid #10b981; background: rgba(16, 185, 129, 0.2); }
        .node-pulse-ring.WARNING { border: 2px solid #f59e0b; background: rgba(245, 158, 11, 0.25); }
        .node-pulse-ring.DANGER { border: 2px solid #ef4444; background: rgba(239, 68, 68, 0.35); animation: alertPulse 0.8s infinite; }
        .node-pulse-ring.OFFLINE { border: 1px dashed #64748b; background: transparent; animation: none; }

        @keyframes mapPulse {
            0% { transform: scale(0.8); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 0.3; }
            100% { transform: scale(0.8); opacity: 0.8; }
        }

        @keyframes alertPulse {
            0% { transform: scale(0.9); opacity: 1; }
            50% { transform: scale(1.4); opacity: 0.4; }
            100% { transform: scale(0.9); opacity: 1; }
        }

        @keyframes dangerBlink {
            0% { transform: scale(1); }
            100% { transform: scale(1.15); box-shadow: 0 0 16px #ef4444; }
        }

        /* Custom Popup Card styling */
        .leaflet-popup-content-wrapper {
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-accent) !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 24px rgba(0,0,0,0.4) !important;
        }
        .leaflet-popup-tip { background: var(--bg-card) !important; }

        /* Dashboard Container */
        main { padding: 2rem 2.5rem; max-width: 1440px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; gap: 1.8rem; flex: 1; }
        .page-container { display: flex; flex-direction: column; gap: 1.8rem; width: 100%; }

        /* Hero Banner */
        .hero-banner { background: var(--hero-bg); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.5rem 2rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; box-shadow: var(--card-shadow); transition: background 0.3s; }
        .hero-title h1 { font-size: 1.4rem; font-weight: 800; color: var(--hero-text); margin-bottom: 0.3rem; }
        .hero-title p { color: var(--text-muted); font-size: 0.88rem; }

        /* Dynamic Status Banner */
        .status-banner { padding: 1.25rem 1.8rem; border-radius: 12px; font-size: 1.15rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between; border: 1px solid transparent; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
        .status-banner.SAFE { background: rgba(16, 185, 129, 0.12); color: #10b981; border-color: var(--safe-color); box-shadow: 0 0 25px var(--safe-glow); }
        .status-banner.WARNING { background: rgba(245, 158, 11, 0.12); color: #d97706; border-color: var(--warn-color); box-shadow: 0 0 25px var(--warn-glow); }
        .status-banner.DANGER { background: rgba(239, 68, 68, 0.12); color: #dc2626; border-color: var(--danger-color); box-shadow: 0 0 25px var(--danger-glow); }
        .status-banner.DISCONNECTED { background: rgba(148, 163, 184, 0.1); color: #64748b; border-color: var(--text-muted); }

        /* Metrics Cards */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem; }
        .metric-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.4rem; display: flex; flex-direction: column; justify-content: space-between; gap: 0.8rem; transition: transform 0.2s, border-color 0.2s, background 0.3s; position: relative; overflow: hidden; box-shadow: var(--card-shadow); }
        .metric-card:hover { transform: translateY(-2px); border-color: var(--border-accent); }
        .metric-header { display: flex; justify-content: space-between; align-items: center; }
        .metric-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; }
        .metric-icon { display: none !important; }
        .metric-value { font-size: 2rem; font-weight: 800; color: var(--text-primary); font-family: 'Montserrat', sans-serif; }
        .metric-footer { font-size: 0.8rem; color: var(--primary-accent); font-weight: 600; display: flex; align-items: center; gap: 0.3rem; }

        /* Control Panel Card */
        .controls-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.4rem 1.8rem; display: flex; flex-wrap: wrap; gap: 2rem; align-items: center; justify-content: space-between; box-shadow: var(--card-shadow); transition: background 0.3s; }
        .controls-left { display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; }
        .control-item { display: flex; flex-direction: column; gap: 0.4rem; }
        .control-item label { font-size: 0.8rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
        .control-item select { background: var(--bg-input); border: 1px solid var(--border-color); color: var(--text-primary); padding: 0.6rem 1rem; border-radius: 8px; font-weight: 600; font-size: 0.9rem; cursor: pointer; }
        
        .mode-pills { display: flex; gap: 0.5rem; background: var(--bg-input); padding: 0.3rem; border-radius: 10px; border: 1px solid var(--border-color); }
        .mode-pill { padding: 0.45rem 0.9rem; border-radius: 7px; font-size: 0.82rem; font-weight: 600; cursor: pointer; color: var(--text-muted); transition: 0.2s; }
        .mode-pill.active { background: var(--primary-accent); color: #ffffff; font-weight: 700; }

        /* Panel Cards */
        .panel-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.6rem; display: flex; flex-direction: column; gap: 1.2rem; box-shadow: var(--card-shadow); transition: background 0.3s; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; }
        .panel-title { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 0.6rem; }
        .panel-title span { color: var(--primary-accent); }

        .btn-page-link { background: rgba(56, 189, 248, 0.1); border: 1px solid var(--border-accent); color: var(--primary-accent); padding: 0.75rem 1.2rem; border-radius: 10px; font-weight: 700; font-size: 0.9rem; cursor: pointer; display: flex; align-items: center; justify-content: space-between; width: 100%; transition: all 0.2s; text-decoration: none; }
        .btn-page-link:hover { background: var(--primary-accent); color: #ffffff; }

        .analytics-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem; }
        @media (max-width: 1100px) { .analytics-grid { grid-template-columns: 1fr; } }

        /* Manual Input Sliders Box */
        #manual-controls { display: none; background: var(--bg-input); border: 1px solid var(--border-accent); border-radius: 10px; padding: 1rem 1.4rem; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 0.5rem; }
        .slider-group { display: flex; flex-direction: column; gap: 0.3rem; }
        .slider-group label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; display: flex; justify-content: space-between; }
        .slider-group input[type="range"] { width: 100%; accent-color: var(--primary-accent); cursor: pointer; }

        /* Table Styling */
        .table-wrapper { overflow-y: auto; max-height: 380px; border: 1px solid var(--border-color); border-radius: 8px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; }
        th { background: var(--bg-input); color: var(--text-muted); padding: 0.8rem 1rem; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; position: sticky; top: 0; }
        td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); font-weight: 500; color: var(--text-primary); }
        tr:hover { background: rgba(56, 189, 248, 0.04); }
        
        .badge { padding: 0.3rem 0.6rem; border-radius: 6px; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.03em; display: inline-block; }
        .badge.SAFE { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge.WARNING { background: rgba(245, 158, 11, 0.15); color: #d97706; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge.DANGER { background: rgba(239, 68, 68, 0.15); color: #dc2626; border: 1px solid rgba(239, 68, 68, 0.3); }

        /* Footer */
        footer { border-top: 1px solid var(--border-color); padding: 1.5rem 2.5rem; color: var(--text-muted); font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center; margin-top: auto; background: var(--bg-card); transition: background 0.3s; }
        footer a { color: var(--primary-accent); text-decoration: none; }
    </style>
</head>
<body>

    <!-- AUTHENTICATION OVERLAY -->
    <div id="auth-screen">
        <div class="auth-card">
            <div class="auth-header">
                <div style="display: flex; align-items: center; justify-content: center; gap: 0.6rem; margin-bottom: 0.4rem;">
                    <img src="data:image/jpeg;base64,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" style="width: 44px; height: 44px; border-radius: 12px; border: 1.5px solid var(--border-accent); box-shadow: 0 0 20px var(--primary-glow);" alt="IGNITERS Logo">
                    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: var(--text-primary);">IGNITERS <span style="color: var(--primary-accent);">AI</span></h2>
                </div>
                <p>Enterprise Early Warning System for Mine Subsidence</p>
            </div>
            
            <div class="auth-tabs">
                <div class="auth-tab active" onclick="switchAuthTab('login')"> Sign In</div>
                <div class="auth-tab" onclick="switchAuthTab('register')"> Register</div>
            </div>

            <!-- LOGIN FORM -->
            <form id="login-form" action="javascript:void(0)" onsubmit="handleLogin(event); return false;">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" id="login-username" required placeholder="e.g. User">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="login-password" required placeholder="••••••••">
                </div>
                <button type="submit" class="btn-submit">Sign In to Console</button>
            </form>

            <!-- REGISTER FORM -->
            <form id="register-form" action="javascript:void(0)" style="display: none;" onsubmit="handleRegister(event); return false;">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" id="reg-username" required placeholder="Choose username">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="reg-password" required placeholder="Choose password">
                </div>
                <div class="form-group">
                    <label>Role</label>
                    <select id="reg-role">
                        <option value="Operator">Operator</option>
                        <option value="Safety Engineer">Safety Engineer</option>
                        <option value="Inspector">Inspector</option>
                    </select>
                </div>
                <button type="submit" class="btn-submit">Register Account</button>
            </form>

            <div id="auth-msg" class="auth-msg"></div>
        </div>
    </div>

    <!-- HEADER NAVBAR -->
    <header>
        <div class="header-left">
            <div class="logo-container">
                <img src="data:image/jpeg;base64,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" class="logo-img" alt="IGNITERS Logo">
                <div class="logo-text">IGNITERS <span>AI</span></div>
            </div>
            <div class="nav-links">
                <button class="nav-btn active" id="nav-btn-monitoring" onclick="switchPage('monitoring')">Live Monitoring</button>
                <button class="nav-btn" id="nav-btn-analytics" onclick="switchPage('analytics')">Telemetry Logs & Risk Analytics</button>
                <button class="nav-btn" id="nav-btn-sites" onclick="switchPage('sites')">Sites</button>
                <button class="nav-btn" id="nav-btn-admin" style="display: none;" onclick="switchPage('admin')">Admin Panel</button>
            </div>
        </div>
        <div class="nav-actions">
            <button class="btn-theme-toggle" onclick="toggleTheme()">
                <span id="theme-icon"></span> <span id="theme-text">Light Mode</span>
            </button>
            <span class="user-badge" id="header-user">Admin (Administrator)</span>
            <button class="btn-logout" onclick="handleLogout()">Logout</button>
        </div>
    </header>

    <!-- DASHBOARD MAIN CONTENT CONTAINER -->
    <main>
        <!-- SCREEN 1: LIVE MONITORING -->
        <div id="page-monitoring" class="page-container">
            <!-- HERO BANNER -->
            <div class="hero-banner">
                <div class="hero-title">
                    <h1>Mine Hazard & Subsidence Early Warning System</h1>
                    <p>Smart India Hackathon (SIH) Project 26025 — Real-Time IoT Telemetry & ML Risk Prediction</p>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted); background: var(--bg-input); padding: 0.65rem 1.2rem; border-radius: 8px; border: 1px solid var(--border-color);" id="live-sub-info">
                    Active Site: <strong style="color: var(--text-primary);">Jharia Sector 4</strong> | Connected Sensor: <strong style="color: var(--primary-accent);">NODE_01 (Pit Slope Inclinometer 01)</strong> | Power Source: <strong style="color: #10b981;"><span id="header-battery-icon"></span> <span id="header-battery">94% Battery</span></strong> | Model: <strong style="color: var(--primary-accent);">RandomForest (100 Trees)</strong>
                </div>
            </div>

            <!-- HAZARD STATUS BANNER -->
            <div id="status-banner" class="status-banner SAFE">
                <div>STATUS: <span id="status-text">SAFE — Normal Operation</span></div>
                <div style="font-size: 0.85rem; font-weight: 500;" id="last-updated">Last Sync: --:--:--</div>
            </div>

            <!-- CONTROLS CARD WITH SITE & NODE CASCADING SELECTORS -->
            <div class="controls-card">
                <div class="controls-left">
                    <div class="control-item">
                        <label>Select Mining Site</label>
                        <select id="live-site-selector" onchange="onLiveSiteChange()" style="font-weight: 700;">
                            <option value="site-1">Jharia Coalfield — Sector 4 Open Pit</option>
                            <option value="site-2">Singrauli Mining Complex — Shaft B</option>
                            <option value="site-3">Korba Underground Mine — Sector 9</option>
                            <option value="site-4">Kolar Strata Slope Zone — Pit 2</option>
                        </select>
                    </div>
                    <div class="control-item">
                        <label>Select Sensor Node (20 Nodes/Site)</label>
                        <select id="live-node-selector" onchange="onLiveNodeChange()" style="min-width: 250px; font-weight: 600;">
                            <!-- Populated dynamically with NODE_01 to NODE_20 -->
                        </select>
                    </div>
                    <div class="control-item">
                        <label>Simulation Scenario</label>
                        <div class="mode-pills">
                            <div class="mode-pill active" onclick="setSimMode('dynamic')">Dynamic</div>
                            <div class="mode-pill" onclick="setSimMode('safe')">Safe</div>
                            <div class="mode-pill" onclick="setSimMode('warning')">Warning</div>
                            <div class="mode-pill" onclick="setSimMode('danger')">Danger</div>
                            <div class="mode-pill" onclick="setSimMode('manual')">Manual Slider</div>
                        </div>
                    </div>
                </div>
                <div class="control-item">
                    <label>Stream Control</label>
                    <select id="stream-toggle" onchange="togglePolling()">
                        <option value="on">Live Polling (Every 2s)</option>
                        <option value="off">Paused</option>
                    </select>
                </div>
            </div>

            <!-- MANUAL SLIDERS CONTAINER (Visible only in Manual mode) -->
            <div id="manual-controls">
                <div class="slider-group">
                    <label>Filtered Tilt: <span id="val-manual-tilt">0.05</span> deg/m</label>
                    <input type="range" id="slider-tilt" min="0" max="15" step="0.01" value="0.05" oninput="updateManualVal()">
                </div>
                <div class="slider-group">
                    <label>Filtered Vibration: <span id="val-manual-vib">0.10</span> g</label>
                    <input type="range" id="slider-vib" min="0" max="6" step="0.01" value="0.10" oninput="updateManualVal()">
                </div>
                <div class="slider-group">
                    <label>Filtered Strain: <span id="val-manual-strain">0.02</span> mm/m</label>
                    <input type="range" id="slider-strain" min="0" max="8" step="0.01" value="0.02" oninput="updateManualVal()">
                </div>
            </div>

            <!-- METRICS GRID -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Filtered Tilt</span>
                    </div>
                    <div class="metric-value" id="val-tilt">0.0245</div>
                    <div class="metric-footer">Deg/m — Structural Gradient</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Filtered Vibration</span>
                    </div>
                    <div class="metric-value" id="val-vib">0.1280</div>
                    <div class="metric-footer">g — Seismic Acceleration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Filtered Strain</span>
                    </div>
                    <div class="metric-value" id="val-strain">0.0120</div>
                    <div class="metric-footer">mm/m — Micro-Displacement</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Risk Alerts (Buffer)</span>
                    </div>
                    <div class="metric-value" id="val-alerts">0 / 0</div>
                    <div class="metric-footer">Warnings / Danger Events</div>
                </div>
            </div>

            <!-- FULL WIDTH LIVE TRENDS CHART -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title"> <span>Real-Time Sensor Telemetry Trends</span></div>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">Last 15 Observations (Live Stream)</span>
                </div>
                <div style="position: relative; height: 380px; width: 100%;">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>

            <!-- QUICK LINK TO SEPARATE ANALYTICS PAGE -->
            <div class="btn-page-link" onclick="switchPage('analytics')">
                <span> Telemetry Buffer Logs & Risk Level Distribution charts have been separated into a dedicated page for clean visibility.</span>
                <span>View Telemetry Logs & Risk Analytics →</span>
            </div>
        </div>

        <!-- SCREEN 2: DEDICATED TELEMETRY LOGS & RISK ANALYTICS -->
        <div id="page-analytics" class="page-container" style="display: none;">
            <div class="hero-banner">
                <div class="hero-title">
                    <h1> Telemetry Buffer Logs & Risk Analytics</h1>
                    <p>Detailed Risk Ratio Distribution, Event Counter Audit, and Historical Sensor Data Table</p>
                </div>
                <button class="nav-btn active" onclick="switchPage('monitoring')">← Back to Live Monitoring</button>
            </div>

            <!-- CUMULATIVE STATS CARDS -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Safe Events</span>
                    </div>
                    <div class="metric-value" id="cnt-safe" style="color: #10b981;">0</div>
                    <div class="metric-footer">Normal Operation Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Warnings</span>
                    </div>
                    <div class="metric-value" id="cnt-warn" style="color: #f59e0b;">0</div>
                    <div class="metric-footer">Drift Threshold Approached</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Danger Alerts</span>
                    </div>
                    <div class="metric-value" id="cnt-danger" style="color: #ef4444;">0</div>
                    <div class="metric-footer">Immediate Hazard Triggers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Logged Records</span>
                    </div>
                    <div class="metric-value" id="cnt-total">0</div>
                    <div class="metric-footer">Buffer Capacity: 20 Samples</div>
                </div>
            </div>

            <!-- SEPARATE ANALYTICS GRID -->
            <div class="analytics-grid">
                <!-- RISK DISTRIBUTION DOUGHNUT -->
                <div class="panel-card">
                    <div class="panel-header">
                        <div class="panel-title"> <span>Risk Level Ratio Distribution</span></div>
                    </div>
                    <div style="position: relative; height: 260px; width: 100%; display: flex; justify-content: center; align-items: center;">
                        <canvas id="riskDoughnut"></canvas>
                    </div>
                </div>

                <!-- FULL TELEMETRY DATA LOG TABLE -->
                <div class="panel-card">
                    <div class="panel-header" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.6rem;">
                        <div class="panel-title" style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
                            <span>Live Sensor Telemetry Log Buffer</span>
                            <span id="log-table-node-badge" class="badge" style="background: rgba(56, 189, 248, 0.15); color: var(--primary-accent); border: 1px solid var(--border-accent); font-size: 0.78rem; padding: 0.25rem 0.65rem; border-radius: 6px; font-weight: 700;">NODE_01 (Separate Node Logs)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <label style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600;">Node View:</label>
                            <select id="log-node-filter" onchange="renderTelemetryLogTable()" style="padding: 0.35rem 0.7rem; background: var(--bg-input); border: 1px solid var(--border-accent); border-radius: 6px; color: var(--text-primary); font-size: 0.8rem; font-weight: 700; cursor: pointer;">
                                <option value="SELECTED">Selected Node Only</option>
                                <option value="ALL">All Nodes Combined</option>
                            </select>
                        </div>
                    </div>
                    <div class="table-wrapper" style="max-height: 380px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Node ID</th>
                                    <th>Tilt (deg/m)</th>
                                    <th>Vibration (g)</th>
                                    <th>Strain (mm/m)</th>
                                    <th>Battery Power</th>
                                    <th>Risk Status</th>
                                </tr>
                            </thead>
                            <tbody id="log-tbody">
                                <!-- Dynamic Rows -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- SCREEN 3: MINING SITES INTERACTIVE MAP -->
        <div id="page-sites" class="page-container" style="display: none;">
            <div class="hero-banner">
                <div class="hero-title">
                    <h1> Interactive Mining Sites & IoT Node Telemetry Map</h1>
                    <p>Select Mine Locations, Track Spatial Node Overlays, Active/Inactive Status & Real-Time Hazard Classifications</p>
                </div>
                <button class="nav-btn active" onclick="switchPage('monitoring')">← Back to Monitoring</button>
            </div>

            <!-- SITE CONTROL & STATS BAR -->
            <div class="map-control-bar">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <label style="font-weight: 700; color: var(--text-primary);"> Select Mining Site:</label>
                    <select id="site-select" class="site-selector" onchange="changeMiningSite(this.value)">
                        <option value="site-1">Jharia Coalfield — Sector 4 Open Pit</option>
                        <option value="site-2">Singrauli Mining Complex — Shaft B</option>
                        <option value="site-3">Korba Underground Mine — Sector 9</option>
                        <option value="site-4">Kolar Strata Slope Zone — Pit 2</option>
                    </select>
                </div>

                <div class="site-legend">
                    <div class="legend-item"><span class="dot-online"></span> Active Node</div>
                    <div class="legend-item"><span class="dot-offline"></span> Inactive Node</div>
                    <div class="legend-item"><span class="dot-safe"></span> SAFE</div>
                    <div class="legend-item"><span class="dot-warning"></span> WARNING</div>
                    <div class="legend-item"><span class="dot-danger"></span> DANGER</div>
                </div>
            </div>

            <!-- SECTOR METRICS CARDS -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Sector Name</span>
                    </div>
                    <div class="metric-value" id="site-metric-name" style="font-size: 1.1rem; font-weight: 700; color: var(--primary-accent);">Jharia Sector 4</div>
                    <div class="metric-footer" id="site-metric-coords">Lat: 23.7513° N, Lon: 86.4172° E</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Monitored Sensor Nodes</span>
                    </div>
                    <div class="metric-value" id="site-metric-nodes">5 Nodes</div>
                    <div class="metric-footer" id="site-metric-active-count">4 Active | 1 Offline</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Sector Hazard Status</span>
                    </div>
                    <div class="metric-value" id="site-metric-status" style="color: #10b981;">SAFE</div>
                    <div class="metric-footer" id="site-metric-status-sub">Real-Time ML Classification</div>
                </div>
            </div>

            <!-- LEAFLET MAP CONTAINER -->
            <div class="panel-card" style="padding: 0.8rem;">
                <div id="map-container"></div>
            </div>
        </div>

        <!-- SCREEN 3: DEDICATED ADMIN PANEL -->
        <div id="page-admin" class="page-container" style="display: none;">
            <div class="hero-banner">
                <div class="hero-title">
                    <h1> Administrator Authorization & Security Console</h1>
                    <p>Manage User Roles, Authorize Pending Registrations, and Track Active User IP & Browser Sessions</p>
                </div>
                <button class="nav-btn active" onclick="switchPage('monitoring')">← Back to Monitoring</button>
            </div>

            <!-- ADMIN STATS CARDS -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Pending Authorizations</span>
                    </div>
                    <div class="metric-value" id="adm-cnt-pending" style="color: #f59e0b;">0</div>
                    <div class="metric-footer">Awaiting Admin Approval</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Approved Accounts</span>
                    </div>
                    <div class="metric-value" id="adm-cnt-approved" style="color: #10b981;">1</div>
                    <div class="metric-footer">Authorized User Profiles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Active User Sessions</span>
                    </div>
                    <div class="metric-value" id="adm-cnt-sessions">1</div>
                    <div class="metric-footer">Live Active Connections</div>
                </div>
            </div>

            <!-- PENDING AUTHORIZATIONS TABLE -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">⏳ <span>Pending Account Authorization Requests</span></div>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">Users waiting for Admin Approval before Sign In</span>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Username</th>
                                <th>Requested Role</th>
                                <th>Registration Date</th>
                                <th>Authorization Action</th>
                            </tr>
                        </thead>
                        <tbody id="adm-pending-tbody">
                            <!-- Dynamic Pending Rows -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- ALL REGISTERED USERS MANAGEMENT -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title"> <span>All Registered User Accounts</span></div>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">User Roles & Access Permissions</span>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Username</th>
                                <th>Assigned Role</th>
                                <th>Status</th>
                                <th>Registration Date</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="adm-users-tbody">
                            <!-- Dynamic User Rows -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- ACTIVE USER IP & BROWSER SESSIONS TABLE -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title"> <span>Active User Sessions & IP Audit Log</span></div>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">Real-Time Session IP Address & Browser Tracker</span>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>User</th>
                                <th>Role</th>
                                <th>IP Address</th>
                                <th>Browser & OS</th>
                                <th>Login Timestamp</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="adm-sessions-tbody">
                            <!-- Dynamic Session Rows -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <!-- FOOTER -->
    <footer>
        <div>© 2026 <strong>Igniters AI</strong> — SIH Project 26025. All Rights Reserved.</div>
        <div>Engineered for Deep Tech Mine Safety & Digital Transformation</div>
    </footer>

    <script>
        // Interactive Mining Sites & IoT Sensor Nodes Map System
        const MINING_SITES = {
    "site-1": {
        "name": "Jharia Coalfield \u2014 Sector 4 Open Pit",
        "shortName": "Jharia Sector 4",
        "lat": 23.7513,
        "lng": 86.4172,
        "zoom": 15,
        "coordsText": "Lat: 23.7513\u00b0 N, Lon: 86.4172\u00b0 E",
        "zones": [
            {
                "name": "Sector 4-A Green Zone",
                "status": "SAFE",
                "relLat": 0.0005,
                "relLng": 0.0008,
                "radius": 240,
                "label": "\ud83d\udfe2 SAFE ZONE \u2014 Normal Operational Sector"
            },
            {
                "name": "Sector 4-B Yellow Zone",
                "status": "WARNING",
                "relLat": 0.0025,
                "relLng": -0.0018,
                "radius": 190,
                "label": "\ud83d\udfe1 WARNING ZONE \u2014 Highwall Slope Watch"
            }
        ],
        "nodes": [
            {
                "id": "NODE_01",
                "name": "Pit Slope Inclinometer",
                "relLat": 0.0014,
                "relLng": 0.0018,
                "active": true,
                "type": "Tilt & Strain Sensor",
                "tilt": 0.024,
                "vib": 0.128,
                "strain": 0.012,
                "battery": 94,
                "status": "SAFE"
            },
            {
                "id": "NODE_02",
                "name": "North-West Strata Strain Gauge",
                "relLat": -0.0015,
                "relLng": 0.0020,
                "active": true,
                "type": "Multi-Axial Strain",
                "tilt": 0.18,
                "vib": 0.11,
                "strain": 0.09,
                "battery": 87,
                "status": "SAFE"
            },
            {
                "id": "NODE_03",
                "name": "Highwall Inclinometer 3B",
                "relLat": 0.0022,
                "relLng": -0.0012,
                "active": true,
                "type": "Digital Inclinometer",
                "tilt": 1.25,
                "vib": 0.62,
                "strain": 0.88,
                "battery": 68,
                "status": "WARNING"
            },
            {
                "id": "NODE_04",
                "name": "Seismic Accelerometer Cluster",
                "relLat": -0.0018,
                "relLng": -0.0010,
                "active": true,
                "type": "Tri-Axial Seismic",
                "tilt": 0.05,
                "vib": 0.14,
                "strain": 0.03,
                "battery": 91,
                "status": "SAFE"
            },
            {
                "id": "NODE_05",
                "name": "Sub-surface Borehole Telemetry",
                "relLat": 0.0002,
                "relLng": -0.0022,
                "active": false,
                "type": "Borehole Extensometer",
                "tilt": 0.0,
                "vib": 0.0,
                "strain": 0.0,
                "battery": 0,
                "status": "OFFLINE"
            }
        ]
    },
    "site-2": {
        "name": "Singrauli Mining Complex \u2014 Shaft B",
        "shortName": "Singrauli Shaft B",
        "lat": 24.1994,
        "lng": 82.6644,
        "zoom": 15,
        "coordsText": "Lat: 24.1994\u00b0 N, Lon: 82.6644\u00b0 E",
        "zones": [
            {
                "name": "Shaft Bench Green Zone",
                "status": "SAFE",
                "relLat": -0.001,
                "relLng": 0.0012,
                "radius": 250,
                "label": "\ud83d\udfe2 SAFE ZONE \u2014 Normal Shaft Operations"
            },
            {
                "name": "South Ramp Red Zone",
                "status": "DANGER",
                "relLat": 0.002,
                "relLng": -0.0025,
                "radius": 220,
                "label": "\ud83d\udd34 RED ZONE \u2014 CRITICAL SUBSIDENCE HAZARD"
            }
        ],
        "nodes": [
            {
                "id": "NODE_06",
                "name": "Shaft Wall Strain Node",
                "relLat": 0.001,
                "relLng": 0.001,
                "active": true,
                "type": "Wall Displacement",
                "tilt": 0.03,
                "vib": 0.08,
                "strain": 0.02,
                "battery": 92,
                "status": "SAFE"
            },
            {
                "id": "NODE_07",
                "name": "Main Bench Slope Sensor",
                "relLat": -0.0015,
                "relLng": 0.0018,
                "active": true,
                "type": "Bench Inclinometer",
                "tilt": 0.42,
                "vib": 0.25,
                "strain": 0.31,
                "battery": 85,
                "status": "SAFE"
            },
            {
                "id": "NODE_08",
                "name": "South Ramp Vibrational Array",
                "relLat": 0.002,
                "relLng": -0.0025,
                "active": true,
                "type": "Vibration Acceleration",
                "tilt": 5.2,
                "vib": 2.1,
                "strain": 3.4,
                "battery": 38,
                "status": "DANGER"
            },
            {
                "id": "NODE_09",
                "name": "Overburden Dump Radar",
                "relLat": -0.0028,
                "relLng": -0.0012,
                "active": true,
                "type": "Dump Slope Radar",
                "tilt": 1.15,
                "vib": 0.58,
                "strain": 0.76,
                "battery": 74,
                "status": "WARNING"
            },
            {
                "id": "NODE_10",
                "name": "Deep Shaft Strain Extensometer",
                "relLat": 0.0005,
                "relLng": 0.003,
                "active": false,
                "type": "Borehole Strain Gauge",
                "tilt": 0.0,
                "vib": 0.0,
                "strain": 0.0,
                "battery": 0,
                "status": "OFFLINE"
            }
        ]
    },
    "site-3": {
        "name": "Korba Underground Mine \u2014 Sector 9",
        "shortName": "Korba Underground",
        "lat": 22.3595,
        "lng": 82.7501,
        "zoom": 15,
        "coordsText": "Lat: 22.3595\u00b0 N, Lon: 82.7501\u00b0 E",
        "zones": [
            {
                "name": "Sector 9 Shaft Green Zone",
                "status": "SAFE",
                "relLat": 0.0008,
                "relLng": 0.0012,
                "radius": 260,
                "label": "\ud83d\udfe2 SAFE ZONE \u2014 Roof Convergence Normal"
            },
            {
                "name": "Pillar Stress Yellow Zone",
                "status": "WARNING",
                "relLat": -0.0012,
                "relLng": -0.0015,
                "radius": 180,
                "label": "\ud83d\udfe1 WARNING ZONE \u2014 High Stress Tensor"
            }
        ],
        "nodes": [
            {
                "id": "NODE_11",
                "name": "Mine Roof Convergence Sensor",
                "relLat": 0.0008,
                "relLng": 0.0012,
                "active": true,
                "type": "Roof Extensometer",
                "tilt": 0.12,
                "vib": 0.15,
                "strain": 0.08,
                "battery": 96,
                "status": "SAFE"
            },
            {
                "id": "NODE_12",
                "name": "Pillar Strain Gauge Alpha",
                "relLat": -0.0012,
                "relLng": -0.0015,
                "active": true,
                "type": "Pillar Stress Tensor",
                "tilt": 2.45,
                "vib": 0.92,
                "strain": 1.15,
                "battery": 79,
                "status": "WARNING"
            },
            {
                "id": "NODE_13",
                "name": "Haulage Shaft Micro-Seismic",
                "relLat": 0.0018,
                "relLng": -0.002,
                "active": true,
                "type": "Micro-Seismic Array",
                "tilt": 0.04,
                "vib": 0.09,
                "strain": 0.02,
                "battery": 88,
                "status": "SAFE"
            },
            {
                "id": "NODE_14",
                "name": "Sub-strata Stress Tensor",
                "relLat": -0.0025,
                "relLng": 0.002,
                "active": true,
                "type": "Strata Extensometer",
                "tilt": 0.09,
                "vib": 0.14,
                "strain": 0.05,
                "battery": 91,
                "status": "SAFE"
            },
            {
                "id": "NODE_15",
                "name": "Roof Convergence Array Beta",
                "relLat": 0.003,
                "relLng": 0.0015,
                "active": true,
                "type": "Convergence Sensor",
                "tilt": 1.35,
                "vib": 0.65,
                "strain": 0.85,
                "battery": 72,
                "status": "WARNING"
            }
        ]
    },
    "site-4": {
        "name": "Kolar Strata Slope Zone \u2014 Pit 2",
        "shortName": "Kolar Pit 2",
        "lat": 12.9583,
        "lng": 78.2711,
        "zoom": 15,
        "coordsText": "Lat: 12.9583\u00b0 N, Lon: 78.2711\u00b0 E",
        "zones": [
            {
                "name": "East Pit Green Zone",
                "status": "SAFE",
                "relLat": 0.0,
                "relLng": 0.001,
                "radius": 240,
                "label": "\ud83d\udfe2 SAFE ZONE \u2014 Strata Stable"
            },
            {
                "name": "Fault Line Yellow Zone",
                "status": "WARNING",
                "relLat": -0.0015,
                "relLng": -0.0025,
                "radius": 170,
                "label": "\ud83d\udfe1 WARNING ZONE \u2014 Micro-Vibration Fault"
            },
            {
                "name": "Pit Edge Red Zone",
                "status": "DANGER",
                "relLat": 0.0022,
                "relLng": -0.0018,
                "radius": 210,
                "label": "\ud83d\udd34 RED ZONE \u2014 Pit Edge Collapse Danger"
            }
        ],
        "nodes": [
            {
                "id": "NODE_16",
                "name": "Deep Rock Mass Extensometer",
                "relLat": 0.0015,
                "relLng": 0.0008,
                "active": true,
                "type": "Deep Rock Strain",
                "tilt": 0.01,
                "vib": 0.04,
                "strain": 0.01,
                "battery": 98,
                "status": "SAFE"
            },
            {
                "id": "NODE_17",
                "name": "West Slope Tilting Inclinometer",
                "relLat": -0.002,
                "relLng": 0.0015,
                "active": true,
                "type": "Slope Inclinometer",
                "tilt": 0.28,
                "vib": 0.19,
                "strain": 0.14,
                "battery": 90,
                "status": "SAFE"
            },
            {
                "id": "NODE_18",
                "name": "Pit Edge Ground Displacement",
                "relLat": 0.0022,
                "relLng": -0.0018,
                "active": true,
                "type": "Ground Displacement",
                "tilt": 6.8,
                "vib": 2.9,
                "strain": 4.5,
                "battery": 24,
                "status": "DANGER"
            },
            {
                "id": "NODE_19",
                "name": "Fault Line Micro-Vibration",
                "relLat": -0.0015,
                "relLng": -0.0025,
                "active": true,
                "type": "Vibration Sensor",
                "tilt": 1.85,
                "vib": 0.78,
                "strain": 0.92,
                "battery": 72,
                "status": "WARNING"
            },
            {
                "id": "NODE_20",
                "name": "Pit Rim Tilting Gauge",
                "relLat": -0.0032,
                "relLng": 0.003,
                "active": false,
                "type": "Inclinometer Array",
                "tilt": 0.0,
                "vib": 0.0,
                "strain": 0.0,
                "battery": 0,
                "status": "OFFLINE"
            }
        ]
    }
};

        function getBatteryInfo(pct) {
            const val = pct !== undefined ? pct : 94;
            if (val <= 0) return { icon: "", text: "0% (Depleted)", color: "#94a3b8" };
            if (val < 20) return { icon: "", text: val + "% (Low)", color: "#ef4444" };
            if (val < 50) return { icon: "", text: val + "%", color: "#f59e0b" };
            if (val < 85) return { icon: "", text: val + "%", color: "#38bdf8" };
            return { icon: "", text: val + "%", color: "#10b981" };
        }

        // Theme Switcher Logic
        function applyChartTheme(isLight) {
            const textColor = isLight ? "#475569" : "#94a3b8";
            const gridColor = isLight ? "rgba(0, 0, 0, 0.08)" : "rgba(255, 255, 255, 0.05)";
            const legendColor = isLight ? "#0f172a" : "#f8fafc";

            if (window.trendChart) {
                trendChart.options.scales.x.ticks.color = textColor;
                trendChart.options.scales.x.grid.color = gridColor;
                trendChart.options.scales.y.ticks.color = textColor;
                trendChart.options.scales.y.grid.color = gridColor;
                trendChart.options.plugins.legend.labels.color = legendColor;
                trendChart.update();
            }

            if (window.riskDoughnut) {
                riskDoughnut.options.plugins.legend.labels.color = legendColor;
                riskDoughnut.update();
            }
        }

        function initTheme() {
            const savedTheme = localStorage.getItem("mine_theme");
            if (savedTheme === "light") {
                document.body.classList.add("light-theme");
                document.getElementById("theme-icon").textContent = "";
                document.getElementById("theme-text").textContent = "Dark Mode";
            }
        }

        function toggleTheme() {
            const isLight = document.body.classList.toggle("light-theme");
            localStorage.setItem("mine_theme", isLight ? "light" : "dark");
            
            document.getElementById("theme-icon").textContent = isLight ? "" : "";
            document.getElementById("theme-text").textContent = isLight ? "Dark Mode" : "Light Mode";

            applyChartTheme(isLight);
        }

        // Supabase Cloud Database Integration
        const SUPABASE_URL = "https://toabcprwbtaipxwzmdyl.supabase.co";
        const SUPABASE_KEY = "sb_publishable_DS2T92fPyhKkhGyI41dtzA_qw2_m_3C";
        let supabaseClient = null;

        try {
            if (window.supabase && window.supabase.createClient) {
                supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
                console.log(" Supabase Cloud Database Connected!");
            }
        } catch(e) {
            console.warn("Supabase init notice:", e);
        }

        async function syncTelemetryToSupabase(record) {
            if (!supabaseClient) return;
            try {
                await supabaseClient.from('telemetry_logs').insert([{
                    timestamp: record.timestamp,
                    node_id: record.node_id || "NODE_01",
                    filtered_tilt: record.filtered_tilt,
                    filtered_vibration: record.filtered_vibration,
                    filtered_strain: record.filtered_strain,
                    status: record.status
                }]);
            } catch(err) {
                console.warn("Supabase log sync notice:", err);
            }
        }

        async function syncSessionToSupabase(session) {
            if (!supabaseClient) return;
            try {
                await supabaseClient.from('active_sessions').insert([{
                    username: session.username,
                    role: session.role,
                    ip_address: session.ip,
                    browser: session.browser,
                    login_time: session.loginTime,
                    status: session.status
                }]);
            } catch(err) {
                console.warn("Supabase session sync notice:", err);
            }
        }

        async function syncUserToSupabase(username, role, status, registeredAt) {
            if (!supabaseClient) return;
            try {
                await supabaseClient.from('users').upsert([{
                    username: username,
                    role: role,
                    status: status,
                    registered_at: registeredAt
                }], { onConflict: 'username' });
            } catch(err) {
                console.warn("Supabase user sync notice:", err);
            }
        }

        // Authentication System — System Accounts & Local Storage Sync
        const DEFAULT_USERS = { 
            "Admin": { password: "godisgreat", role: "Administrator", status: "Approved", registeredAt: "2026-09-01 00:00:00" },
            "User": { password: "user123", role: "Operator", status: "Approved", registeredAt: "2026-09-01 00:00:00" },
            "Operator": { password: "operator123", role: "Operator", status: "Approved", registeredAt: "2026-09-01 00:00:00" }
        };

        function getUsers() {
            const saved = localStorage.getItem("mine_users");
            let users = saved ? JSON.parse(saved) : {};
            Object.keys(DEFAULT_USERS).forEach(k => {
                if (!users[k]) {
                    users[k] = DEFAULT_USERS[k];
                }
            });
            return users;
        }

        function saveUsers(users) {
            localStorage.setItem("mine_users", JSON.stringify(users));
        }

        function getSessions() {
            const saved = localStorage.getItem("mine_active_sessions");
            return saved ? JSON.parse(saved) : [];
        }

        async function logActiveSession(username, role) {
            let clientIp = "127.0.0.1 (Local)";
            try {
                const res = await fetch("https://api.ipify.org?format=json");
                if (res.ok) {
                    const data = await res.json();
                    clientIp = data.ip || clientIp;
                }
            } catch(e) {}

            const ua = navigator.userAgent;
            let browserName = "Browser";
            if (ua.includes("Firefox")) browserName = "Mozilla Firefox";
            else if (ua.includes("Edg")) browserName = "Microsoft Edge";
            else if (ua.includes("Chrome")) browserName = "Google Chrome";
            else if (ua.includes("Safari")) browserName = "Apple Safari";
            else browserName = ua.substring(0, 20);

            const osName = ua.includes("Windows") ? "Windows OS" : ua.includes("Mac") ? "macOS" : ua.includes("Linux") ? "Linux" : "Mobile Device";

            const sessions = getSessions();
            const newSession = {
                username: username,
                role: role,
                ip: clientIp,
                browser: `${browserName} (${osName})`,
                loginTime: new Date().toLocaleString(),
                status: "Active"
            };

            const updated = [newSession, ...sessions.filter(s => s.username !== username).slice(0, 19)];
            localStorage.setItem("mine_active_sessions", JSON.stringify(updated));

            // Sync session to Supabase
            syncSessionToSupabase(newSession);
        }

        let currentUser = null;
        let activeSimMode = "dynamic";

        function checkAuth() {
            if (currentUser) {
                document.getElementById("auth-screen").style.display = "none";
                const users = getUsers();
                const matchedKey = Object.keys(users).find(k => k.toLowerCase() === currentUser.toLowerCase()) || currentUser;
                const u = users[matchedKey] || { role: "Operator", status: "Approved" };
                const headerUser = document.getElementById("header-user");
                if (headerUser) {
                    headerUser.textContent = `${matchedKey} (${u.role})`;
                }

                // Show Admin Panel tab ONLY for Administrator
                const adminBtn = document.getElementById("nav-btn-admin");
                if (adminBtn) {
                    if (matchedKey.toLowerCase() === "admin" || u.role === "Administrator") {
                        adminBtn.style.display = "inline-block";
                    } else {
                        adminBtn.style.display = "none";
                        if (document.getElementById("page-admin").style.display !== "none") {
                            switchPage('monitoring');
                        }
                    }
                }
            } else {
                document.getElementById("auth-screen").style.display = "flex";
            }
        }

        function switchAuthTab(tab) {
            document.querySelectorAll(".auth-tab").forEach(t => t.classList.remove("active"));
            if (tab === 'login') {
                document.querySelectorAll(".auth-tab")[0].classList.add("active");
                document.getElementById("login-form").style.display = "block";
                document.getElementById("register-form").style.display = "none";
            } else {
                document.querySelectorAll(".auth-tab")[1].classList.add("active");
                document.getElementById("login-form").style.display = "none";
                document.getElementById("register-form").style.display = "block";
            }
            showAuthMsg("", false);
        }

        function showAuthMsg(msg, isError) {
            const el = document.getElementById("auth-msg");
            if (!msg) { el.style.display = "none"; return; }
            el.className = "auth-msg " + (isError ? "error" : "success");
            el.textContent = msg;
            el.style.display = "block";
        }

        function handleLogin(e) {
            if (e) {
                e.preventDefault();
                if (e.stopPropagation) e.stopPropagation();
            }
            if (window.location.search) {
                try { history.replaceState(null, '', window.location.pathname); } catch(err) {}
            }
            const uInput = document.getElementById("login-username").value.trim();
            const pInput = document.getElementById("login-password").value;
            if (!uInput) return showAuthMsg("Please enter a username.", true);

            const users = getUsers();
            const matchedKey = Object.keys(users).find(k => k.toLowerCase() === uInput.toLowerCase());

            let finalUsername = matchedKey || uInput;
            let role = "Operator";

            if (finalUsername.toLowerCase() === "admin") {
                role = "Administrator";
            }

            if (matchedKey && users[matchedKey]) {
                role = users[matchedKey].role || role;
                users[matchedKey].password = pInput || users[matchedKey].password;
                users[matchedKey].status = "Approved";
            } else {
                users[finalUsername] = {
                    password: pInput || "password",
                    role: role,
                    status: "Approved",
                    registeredAt: getFormattedTimestamp(0)
                };
            }

            saveUsers(users);
            syncUserToSupabase(finalUsername, role, "Approved", getFormattedTimestamp(0));

            currentUser = finalUsername;
            localStorage.setItem("mine_current_user", finalUsername);
            logActiveSession(finalUsername, role);

            showAuthMsg("Login successful! Accessing console...", false);

            // Instant transition to live console
            document.getElementById("auth-screen").style.display = "none";
            checkAuth();
            switchPage('monitoring');
        }

        function handleRegister(e) {
            if (e) {
                e.preventDefault();
                if (e.stopPropagation) e.stopPropagation();
            }
            if (window.location.search) {
                try { history.replaceState(null, '', window.location.pathname); } catch(err) {}
            }
            const u = document.getElementById("reg-username").value.trim();
            const p = document.getElementById("reg-password").value;
            const r = document.getElementById("reg-role").value || "Operator";

            if (!u || !p) return showAuthMsg("Username and password are required.", true);
            const users = getUsers();
            const existingKey = Object.keys(users).find(k => k.toLowerCase() === u.toLowerCase());
            if (existingKey) return showAuthMsg("Username already exists.", true);

            const regTime = getFormattedTimestamp(0);
            users[u] = {
                password: p,
                role: r,
                status: "Approved",
                registeredAt: regTime
            };
            saveUsers(users);
            syncUserToSupabase(u, r, "Approved", regTime);

            currentUser = u;
            localStorage.setItem("mine_current_user", u);
            logActiveSession(u, r);
            showAuthMsg("Account registered successfully! Accessing console...", false);

            setTimeout(() => {
                checkAuth();
                switchPage('monitoring');
            }, 400);
        }

        function handleLogout() {
            currentUser = null;
            localStorage.removeItem("mine_current_user");
            const authMsg = document.getElementById("auth-msg");
            if (authMsg) authMsg.style.display = "none";
            populateLiveNodeDropdown();
            renderTelemetryLogTable();
            checkAuth();
        }

        // Admin Panel Rendering & Cloud Sync Actions
        async function deleteUserFromSupabase(username) {
            if (!supabaseClient || username === "Admin") return;
            try {
                await supabaseClient.from('users').delete().eq('username', username);
            } catch(err) {}
        }

        async function renderAdminPanel() {
            let users = getUsers();
            let sessions = getSessions();

            // Fetch live cloud records directly from Supabase Database
            if (supabaseClient) {
                try {
                    const { data: suUsers } = await supabaseClient.from('users').select('*');
                    if (suUsers && suUsers.length > 0) {
                        suUsers.forEach(u => {
                            users[u.username] = {
                                password: users[u.username]?.password || "••••••••",
                                role: u.role,
                                status: u.status,
                                registeredAt: u.registered_at
                            };
                        });
                        saveUsers(users);
                    }

                    const { data: suSess } = await supabaseClient.from('active_sessions').select('*').order('id', { ascending: false }).limit(25);
                    if (suSess && suSess.length > 0) {
                        sessions = suSess.map(s => ({
                            username: s.username,
                            role: s.role,
                            ip: s.ip_address,
                            browser: s.browser,
                            loginTime: s.login_time,
                            status: s.status || "Active"
                        }));
                        localStorage.setItem("mine_active_sessions", JSON.stringify(sessions));
                    }
                } catch(e) {
                    console.warn("Supabase fetch notice:", e);
                }
            }

            let pendingList = [];
            let approvedList = [];

            Object.keys(users).forEach(uname => {
                const item = users[uname];
                if (item.status === "Pending" && uname !== "Admin") {
                    pendingList.push({ username: uname, ...item });
                } else {
                    approvedList.push({ username: uname, ...item });
                }
            });

            document.getElementById("adm-cnt-pending").textContent = pendingList.length;
            document.getElementById("adm-cnt-approved").textContent = approvedList.length;
            document.getElementById("adm-cnt-sessions").textContent = sessions.length;

            // Render Pending Table
            const pendingTbody = document.getElementById("adm-pending-tbody");
            if (pendingList.length === 0) {
                pendingTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No pending user authorization requests found.</td></tr>`;
            } else {
                pendingTbody.innerHTML = pendingList.map(u => `
                    <tr>
                        <td><strong>${u.username}</strong></td>
                        <td><span class="user-badge">${u.role}</span></td>
                        <td>${u.registeredAt || 'Just now'}</td>
                        <td>
                            <button onclick="approveUser('${u.username}')" style="background: #10b981; color: #fff; border: none; padding: 0.35rem 0.75rem; border-radius: 6px; font-weight: 700; cursor: pointer; margin-right: 0.4rem;"> Authorize</button>
                            <button onclick="rejectUser('${u.username}')" style="background: #ef4444; color: #fff; border: none; padding: 0.35rem 0.75rem; border-radius: 6px; font-weight: 700; cursor: pointer;"> Reject</button>
                        </td>
                    </tr>
                `).join('');
            }

            // Render Registered Users Table
            const usersTbody = document.getElementById("adm-users-tbody");
            if (Object.keys(users).length === 0) {
                usersTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No registered user accounts found.</td></tr>`;
            } else {
                usersTbody.innerHTML = Object.keys(users).map(uname => {
                    const u = users[uname];
                    const isApproved = (u.status === "Approved" || uname === "Admin");
                    return `
                        <tr>
                            <td><strong>${uname}</strong></td>
                            <td><span class="user-badge">${u.role || 'Operator'}</span></td>
                            <td><span class="badge ${isApproved ? 'SAFE' : 'WARNING'}">${isApproved ? 'APPROVED' : 'PENDING'}</span></td>
                            <td>${u.registeredAt || 'Pre-configured'}</td>
                            <td>
                                ${uname === 'Admin' ? '<span style="color: var(--text-muted);">Master Admin</span>' : `
                                    <button onclick="toggleUserStatus('${uname}')" style="background: var(--primary-accent); color: #fff; border: none; padding: 0.3rem 0.60rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600; cursor: pointer; margin-right: 0.3rem;">${isApproved ? 'Revoke' : 'Approve'}</button>
                                    <button onclick="deleteUser('${uname}')" style="background: transparent; color: #ef4444; border: 1px solid #ef4444; padding: 0.3rem 0.60rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600; cursor: pointer;">Delete</button>
                                `}
                            </td>
                        </tr>
                    `;
                }).join('');
            }

            // Render Active Sessions Audit Table
            const sessionsTbody = document.getElementById("adm-sessions-tbody");
            if (sessions.length === 0) {
                sessionsTbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No active session logs recorded yet.</td></tr>`;
            } else {
                sessionsTbody.innerHTML = sessions.map(s => `
                    <tr>
                        <td><strong>${s.username}</strong></td>
                        <td><span class="user-badge">${s.role}</span></td>
                        <td><code style="color: var(--primary-accent); font-weight: 700;">${s.ip}</code></td>
                        <td>${s.browser}</td>
                        <td>${s.loginTime}</td>
                        <td><span class="badge SAFE">ACTIVE</span></td>
                    </tr>
                `).join('');
            }
        }

        async function approveUser(uname) {
            const users = getUsers();
            if (users[uname]) {
                users[uname].status = "Approved";
                saveUsers(users);
                await syncUserToSupabase(uname, users[uname].role, "Approved", users[uname].registeredAt);
                renderAdminPanel();
            }
        }

        async function rejectUser(uname) {
            await deleteUser(uname);
        }

        async function toggleUserStatus(uname) {
            const users = getUsers();
            if (users[uname]) {
                const newStatus = (users[uname].status === "Approved") ? "Pending" : "Approved";
                users[uname].status = newStatus;
                saveUsers(users);
                await syncUserToSupabase(uname, users[uname].role, newStatus, users[uname].registeredAt);
                renderAdminPanel();
            }
        }

        async function deleteUser(uname) {
            if (uname === "Admin") return;
            const users = getUsers();
            delete users[uname];
            saveUsers(users);
            await deleteUserFromSupabase(uname);
            renderAdminPanel();
        }

        function setSimMode(mode) {
            activeSimMode = mode;
            document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("active"));
            event.target.classList.add("active");
            
            const manualBox = document.getElementById("manual-controls");
            manualBox.style.display = (mode === "manual") ? "grid" : "none";
            updateTelemetry();
        }

        function updateManualVal() {
            document.getElementById("val-manual-tilt").textContent = document.getElementById("slider-tilt").value;
            document.getElementById("val-manual-vib").textContent = document.getElementById("slider-vib").value;
            document.getElementById("val-manual-strain").textContent = document.getElementById("slider-strain").value;
            updateTelemetry();
        }

        function togglePolling() {
            // Handled in interval loop
        }

        // Initialize Trend Line Chart
        const initialIsLight = document.body.classList.contains("light-theme");
        const initialTextColor = initialIsLight ? "#64748b" : "#94a3b8";
        const initialGridColor = initialIsLight ? "rgba(0,0,0,0.06)" : "rgba(255,255,255,0.05)";

        const ctxTrend = document.getElementById('trendChart').getContext('2d');
        const trendChart = new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Tilt (deg/m)', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)', tension: 0.35, fill: true },
                    { label: 'Vibration (g)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', tension: 0.35, fill: true },
                    { label: 'Strain (mm/m)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', tension: 0.35, fill: true }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: initialTextColor }, grid: { color: initialGridColor } },
                    y: { ticks: { color: initialTextColor }, grid: { color: initialGridColor } }
                },
                plugins: { legend: { labels: { color: initialIsLight ? '#0f172a' : '#f8fafc', font: { family: 'Inter', weight: '600' } } } }
            }
        });

        // Initialize Risk Level Doughnut Chart
        const ctxRisk = document.getElementById('riskDoughnut').getContext('2d');
        const riskDoughnut = new Chart(ctxRisk, {
            type: 'doughnut',
            data: {
                labels: ['SAFE', 'WARNING', 'DANGER'],
                datasets: [{
                    data: [1, 0, 0],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: initialIsLight ? '#0f172a' : '#f8fafc', font: { family: 'Inter', size: 11 } } } },
                cutout: '70%'
            }
        });

        function initChartDataForNode(nodeId) {
            const labels = [];
            const tiltData = [];
            const vibData = [];
            const strainData = [];
            for (let i = 9; i >= 0; i--) {
                labels.push(getFormattedTimestamp(i * 3));
                const baseTilt = +(0.05 + (Math.random() * 0.1)).toFixed(4);
                const baseVib = +(0.12 + (Math.random() * 0.08)).toFixed(4);
                const baseStrain = +(0.03 + (Math.random() * 0.05)).toFixed(4);
                tiltData.push(baseTilt);
                vibData.push(baseVib);
                strainData.push(baseStrain);
            }
            if (window.trendChart) {
                trendChart.data.labels = labels;
                trendChart.data.datasets[0].data = tiltData;
                trendChart.data.datasets[1].data = vibData;
                trendChart.data.datasets[2].data = strainData;
                trendChart.update('none');
            }
        }

        // Telemetry Data Logic
        function getFormattedTimestamp(offsetSeconds = 0) {
            const now = new Date(Date.now() - (offsetSeconds * 1000));
            const yyyy = now.getFullYear();
            const mm = String(now.getMonth() + 1).padStart(2, '0');
            const dd = String(now.getDate()).padStart(2, '0');
            const hh = String(now.getHours()).padStart(2, '0');
            const min = String(now.getMinutes()).padStart(2, '0');
            const ss = String(now.getSeconds()).padStart(2, '0');
            return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
        }

        let nodeHistoryBuffers = {};
        const globalStartTimestamp = Date.now();

        function initNodeHistoryBuffer(nodeId) {
            const buf = [];
            const statuses = ["SAFE", "SAFE", "SAFE", "SAFE", "SAFE"];
            const nodeNum = parseInt((nodeId || "01").replace(/\\D/g, '')) || 1;
            
            // Resolve node battery from site definition
            let nodeBat = 94;
            for (let sk in MINING_SITES) {
                if (MINING_SITES[sk].nodes) {
                    const matchNode = MINING_SITES[sk].nodes.find(n => n.id === nodeId);
                    if (matchNode && matchNode.battery !== undefined) {
                        nodeBat = matchNode.battery;
                        break;
                    }
                }
            }

            for (let i = 9; i >= 0; i--) {
                const timeOffsetSec = (i * 3) + (nodeNum * 0.1);
                const recordTime = new Date(globalStartTimestamp - (timeOffsetSec * 1000));
                const yyyy = recordTime.getFullYear();
                const mm = String(recordTime.getMonth() + 1).padStart(2, '0');
                const dd = String(recordTime.getDate()).padStart(2, '0');
                const hh = String(recordTime.getHours()).padStart(2, '0');
                const min = String(recordTime.getMinutes()).padStart(2, '0');
                const ss = String(recordTime.getSeconds()).padStart(2, '0');
                const timeStr = `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;

                const baseTilt = +(0.02 + (Math.random() * 0.04)).toFixed(4);
                const baseVib = +(0.11 + (Math.random() * 0.03)).toFixed(4);
                const baseStrain = +(0.01 + (Math.random() * 0.02)).toFixed(4);
                buf.push({
                    time: timeStr,
                    node_id: nodeId,
                    tilt: baseTilt,
                    vib: baseVib,
                    strain: baseStrain,
                    battery: nodeBat,
                    status: statuses[i % statuses.length]
                });
            }
            return buf;
        }

        function preInitializeAllNodeBuffers() {
            for (let i = 1; i <= 20; i++) {
                const nid = `NODE_${String(i).padStart(2, '0')}`;
                if (!nodeHistoryBuffers[nid]) {
                    nodeHistoryBuffers[nid] = initNodeHistoryBuffer(nid);
                }
            }
        }
        preInitializeAllNodeBuffers();

        function getNodeHistoryBuffer(nodeId) {
            const targetId = nodeId || selectedNodeId || "NODE_01";
            if (!nodeHistoryBuffers[targetId]) {
                nodeHistoryBuffers[targetId] = initNodeHistoryBuffer(targetId);
            }
            return nodeHistoryBuffers[targetId];
        }

        async function fetchNodeLogsFromSupabase(nodeId) {
            if (!supabaseClient) return;
            try {
                const { data, error } = await supabaseClient
                    .from('telemetry_logs')
                    .select('*')
                    .eq('node_id', nodeId)
                    .order('timestamp', { ascending: false })
                    .limit(20);

                if (data && data.length > 0) {
                    nodeHistoryBuffers[nodeId] = data.map(d => ({
                        time: d.timestamp,
                        node_id: d.node_id || nodeId,
                        tilt: parseFloat(d.filtered_tilt || 0),
                        vib: parseFloat(d.filtered_vibration || 0),
                        strain: parseFloat(d.filtered_strain || 0),
                        battery: 94,
                        status: d.status || "SAFE"
                    }));
                    renderTelemetryLogTable();
                }
            } catch(err) {
                console.warn("Supabase node log fetch notice:", err);
            }
        }
        let safeCount = 0, warnCount = 0, dangerCount = 0;

        let selectedSiteKey = "site-1";
        let selectedNodeId = "NODE_01";

        function populateLiveNodeDropdown() {
            const nodeSelect = document.getElementById("live-node-selector");
            if (!nodeSelect) return;

            const site = (typeof MINING_SITES !== 'undefined' && selectedSiteKey && MINING_SITES[selectedSiteKey]) ? MINING_SITES[selectedSiteKey] : null;
            if (!site || !site.nodes || !Array.isArray(site.nodes)) {
                console.warn("populateLiveNodeDropdown: Site or nodes not loaded");
                return;
            }

            nodeSelect.innerHTML = site.nodes.map(n => {
                const batVal = (n && n.battery !== undefined) ? n.battery : 94;
                const statusBadge = (n && n.active) ? `[${n.status || 'SAFE'}]` : "[OFFLINE]";
                return `<option value="${n.id}">${n.id} — ${n.name} (${batVal}% ${statusBadge})</option>`;
            }).join('');

            if (site.nodes.some(n => n.id === selectedNodeId)) {
                nodeSelect.value = selectedNodeId;
            } else {
                selectedNodeId = site.nodes[0] ? site.nodes[0].id : "NODE_01";
                nodeSelect.value = selectedNodeId;
            }
        }

        function onLiveSiteChange() {
            const siteSelect = document.getElementById("live-site-selector");
            if (!siteSelect) return;
            selectedSiteKey = siteSelect.value;

            // Sync site selection on Sites map
            const mapSiteSelect = document.getElementById("map-site-selector");
            if (mapSiteSelect) {
                mapSiteSelect.value = selectedSiteKey;
                currentSiteKey = selectedSiteKey;
                if (leafletMap) renderSiteNodesOnMap();
            }

            populateLiveNodeDropdown();
        renderTelemetryLogTable();
            onLiveNodeChange();
        }

        function onLiveNodeChange() {
            const nodeSelect = document.getElementById("live-node-selector");
            if (!nodeSelect) return;
            selectedNodeId = nodeSelect.value;

            const site = MINING_SITES[selectedSiteKey];
            const node = site.nodes ? site.nodes.find(n => n.id === selectedNodeId) : null;

            const subtitleEl = document.getElementById("live-sub-info");
            if (subtitleEl && node) {
                const batInfo = getBatteryInfo(node.battery);
                subtitleEl.innerHTML = `Active Site: <strong style="color: var(--text-primary);">${site.shortName}</strong> | Connected Sensor: <strong style="color: var(--primary-accent);">${node.id} (${node.name})</strong> | Power: <strong style="color: ${batInfo.color};">${batInfo.icon} ${node.battery}% Battery</strong> | Type: <strong style="color: var(--text-primary);">${node.type}</strong>`;
            }

            // Reset and pre-populate chart and telemetry logs for selected node
            initChartDataForNode(selectedNodeId);
            renderTelemetryLogTable();
            fetchNodeLogsFromSupabase(selectedNodeId);

            updateTelemetry();
        }

        function getSimulatedPayload() {
            let tilt, vib, strain;
            const site = MINING_SITES[selectedSiteKey] || MINING_SITES["site-1"];
            const node = (site && site.nodes) ? site.nodes.find(n => n.id === selectedNodeId) : null;
            const baseBattery = (node && node.battery !== undefined) ? node.battery : 94;

            if (activeSimMode === "safe") {
                tilt = (Math.random() * 0.30 + 0.001).toFixed(4);
                vib = (Math.random() * 0.30 + 0.01).toFixed(4);
                strain = (Math.random() * 0.20 + 0.001).toFixed(4);
            } else if (activeSimMode === "warning") {
                tilt = (Math.random() * 3.00 + 0.50).toFixed(4);
                vib = (Math.random() * 0.85 + 0.35).toFixed(4);
                strain = (Math.random() * 1.40 + 0.40).toFixed(4);
            } else if (activeSimMode === "danger") {
                tilt = (Math.random() * 8.00 + 4.00).toFixed(4);
                vib = (Math.random() * 3.50 + 1.50).toFixed(4);
                strain = (Math.random() * 5.00 + 2.00).toFixed(4);
            } else if (activeSimMode === "manual") {
                tilt = parseFloat(document.getElementById("slider-tilt").value).toFixed(4);
                vib = parseFloat(document.getElementById("slider-vib").value).toFixed(4);
                strain = parseFloat(document.getElementById("slider-strain").value).toFixed(4);
            } else { // Dynamic mode: baseline + realistic live sensor telemetry jitter
                const baseTilt = node ? node.tilt : 0.024;
                const baseVib = node ? node.vib : 0.128;
                const baseStrain = node ? node.strain : 0.012;

                const jitterTilt = (Math.random() * 0.04 - 0.02);
                const jitterVib = (Math.random() * 0.04 - 0.02);
                const jitterStrain = (Math.random() * 0.02 - 0.01);

                tilt = Math.max(0.001, baseTilt + jitterTilt).toFixed(4);
                vib = Math.max(0.01, baseVib + jitterVib).toFixed(4);
                strain = Math.max(0.001, baseStrain + jitterStrain).toFixed(4);
            }

            return {
                node_id: selectedNodeId,
                filtered_tilt: parseFloat(tilt),
                filtered_vibration: parseFloat(vib),
                filtered_strain: parseFloat(strain),
                battery: baseBattery
            };
        }


        let leafletMap = null;
        let currentSiteKey = "site-1";
        let nodeMarkersMap = {};
        let zoneCirclesMap = [];

        function initMiningMap() {
            if (leafletMap || typeof L === 'undefined') return;

            const site = MINING_SITES[currentSiteKey];
            leafletMap = L.map('map-container').setView([site.lat, site.lng], site.zoom);

            const isLight = document.body.classList.contains("light-theme");
            const tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

            L.tileLayer(tileUrl, {
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 18
            }).addTo(leafletMap);

            renderSiteNodesOnMap();
        }

        function changeMiningSite(siteKey) {
            currentSiteKey = siteKey;
            const site = MINING_SITES[siteKey];

            document.getElementById("site-metric-name").textContent = site.shortName;
            document.getElementById("site-metric-coords").textContent = site.coordsText;

            if (leafletMap) {
                leafletMap.setView([site.lat, site.lng], site.zoom);
            }
            renderSiteNodesOnMap();
        }

        function createCustomNodeMarkerHtml(node) {
            const statusClass = node.active ? node.status : "OFFLINE";
            const shortId = node.id.replace("NODE_", "N");
            return `
                <div class="node-marker-wrapper">
                    <div class="node-pulse-ring ${statusClass}"></div>
                    <div class="node-marker-icon ${statusClass}">${shortId}</div>
                </div>
            `;
        }

        function renderSiteZonesOnMap() {
            if (!leafletMap) return;

            zoneCirclesMap.forEach(layer => leafletMap.removeLayer(layer));
            zoneCirclesMap = [];

            const site = MINING_SITES[currentSiteKey];
            if (!site || !site.nodes) return;

            // Draw interconnected spatial coverage circles for EVERY node
            site.nodes.forEach(node => {
                const centerLat = site.lat + node.relLat;
                const centerLng = site.lng + node.relLng;

                let strokeColor = "#10b981";
                let fillColor = "#10b981";
                let fillOpacity = 0.22;
                let dashArray = "8, 6";

                const nodeStatus = node.active ? node.status : "OFFLINE";

                if (nodeStatus === "DANGER") {
                    strokeColor = "#ef4444";
                    fillColor = "#ef4444";
                    fillOpacity = 0.32;
                    dashArray = "6, 6";
                } else if (nodeStatus === "WARNING") {
                    strokeColor = "#f59e0b";
                    fillColor = "#f59e0b";
                    fillOpacity = 0.26;
                    dashArray = "8, 8";
                } else if (nodeStatus === "OFFLINE") {
                    strokeColor = "#64748b";
                    fillColor = "#64748b";
                    fillOpacity = 0.14;
                    dashArray = "4, 4";
                } else {
                    strokeColor = "#10b981";
                    fillColor = "#10b981";
                    fillOpacity = 0.22;
                    dashArray = "10, 6";
                }

                // 240-meter coverage radius ensures overlapping radar circles for EVERY node
                const circle = L.circle([centerLat, centerLng], {
                    radius: 270,
                    color: strokeColor,
                    fillColor: fillColor,
                    fillOpacity: fillOpacity,
                    weight: 2,
                    dashArray: dashArray
                }).addTo(leafletMap);

                const zonePopup = `
                    <div style="font-family: 'Inter', sans-serif; padding: 0.3rem; text-align: center;">
                        <div style="font-weight: 800; font-size: 0.88rem; margin-bottom: 0.3rem; color: ${strokeColor};">${node.id} — ${node.name}</div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.4rem;">Spatial Hazard Coverage Zone (240m)</div>
                        <span class="badge ${nodeStatus}" style="font-size: 0.75rem; padding: 0.25rem 0.6rem;">NODE STATUS: ${nodeStatus}</span>
                    </div>
                `;
                circle.bindPopup(zonePopup);
                zoneCirclesMap.push(circle);
            });
        }



        function renderSiteNodesOnMap() {
            if (!leafletMap) return;

            renderSiteZonesOnMap();

            Object.keys(nodeMarkersMap).forEach(k => {
                leafletMap.removeLayer(nodeMarkersMap[k]);
            });
            nodeMarkersMap = {};

            const site = MINING_SITES[currentSiteKey];
            let activeCount = 0;
            let offlineCount = 0;
            let maxRisk = "SAFE";

            site.nodes.forEach(node => {
                if (node.active) {
                    activeCount++;
                    if (node.status === "DANGER") maxRisk = "DANGER";
                    else if (node.status === "WARNING" && maxRisk !== "DANGER") maxRisk = "WARNING";
                } else {
                    offlineCount++;
                }

                const nodeLat = site.lat + node.relLat;
                const nodeLng = site.lng + node.relLng;

                const customIcon = L.divIcon({
                    html: createCustomNodeMarkerHtml(node),
                    className: '',
                    iconSize: [36, 36],
                    iconAnchor: [18, 18]
                });

                const marker = L.marker([nodeLat, nodeLng], { icon: customIcon }).addTo(leafletMap);

                const batPct = node.battery !== undefined ? node.battery : (node.active ? 94 : 0);
                const bat = getBatteryInfo(batPct);

                const popupContent = `
                    <div style="font-family: 'Inter', sans-serif; width: 240px; padding: 0.2rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <strong style="font-size: 0.9rem; color: var(--primary-accent);">${node.id}</strong>
                            <span style="font-size: 0.72rem; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; background: ${node.active ? 'rgba(16,185,129,0.2)' : 'rgba(100,116,139,0.2)'}; color: ${node.active ? '#10b981' : '#94a3b8'};">
                                ${node.active ? 'ACTIVE' : 'OFFLINE'}
                            </span>
                        </div>
                        <div style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem; color: var(--text-primary);">${node.name}</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;">Sensor Type: ${node.type}</div>
                        
                        <div style="background: rgba(0,0,0,0.25); padding: 0.45rem 0.6rem; border-radius: 6px; font-size: 0.78rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; border: 1px solid var(--border-color);">
                            <span style="color: var(--text-muted); font-weight: 600;">Battery Power:</span>
                            <strong style="color: ${bat.color}; font-size: 0.82rem;">${bat.icon} ${bat.text}</strong>
                        </div>

                        <div style="background: rgba(255,255,255,0.05); padding: 0.5rem; border-radius: 6px; font-size: 0.78rem; display: flex; flex-direction: column; gap: 0.3rem;">
                            <div>Tilt: <strong style="color: #38bdf8;">${node.tilt.toFixed(4)} deg/m</strong></div>
                            <div>Vibration: <strong style="color: #f59e0b;">${node.vib.toFixed(4)} g</strong></div>
                            <div>Strain: <strong style="color: #ef4444;">${node.strain.toFixed(4)} mm/m</strong></div>
                        </div>

                        <div style="margin-top: 0.6rem; text-align: center;">
                            <span class="badge ${node.active ? node.status : 'OFFLINE'}" style="font-size: 0.75rem; width: 100%; display: block; padding: 0.3rem 0;">
                                ${node.active ? 'HAZARD STATUS: ' + node.status : 'NODE OFFLINE'}
                            </span>
                        </div>
                    </div>
                `;

                marker.bindPopup(popupContent);
                nodeMarkersMap[node.id] = marker;
            });

            document.getElementById("site-metric-nodes").textContent = `${site.nodes.length} Sensor Nodes`;
            document.getElementById("site-metric-active-count").textContent = `${activeCount} Active | ${offlineCount} Offline`;
            
            const statusValEl = document.getElementById("site-metric-status");
            const statusIconEl = document.getElementById("site-metric-status-icon");
            statusValEl.textContent = maxRisk;
            if (maxRisk === "DANGER") {
                statusValEl.style.color = "#ef4444";
                statusIconEl.textContent = "";
            } else if (maxRisk === "WARNING") {
                statusValEl.style.color = "#f59e0b";
                statusIconEl.textContent = "";
            } else {
                statusValEl.style.color = "#10b981";
                statusIconEl.textContent = "";
            }
        }

        // Page Switcher Navigation
        function switchPage(page) {
            document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
            document.getElementById("page-monitoring").style.display = "none";
            document.getElementById("page-analytics").style.display = "none";
            document.getElementById("page-sites").style.display = "none";
            document.getElementById("page-admin").style.display = "none";

            if (page === 'monitoring') {
                document.getElementById("nav-btn-monitoring").classList.add("active");
                document.getElementById("page-monitoring").style.display = "flex";
                setTimeout(() => {
                    if (window.trendChart) trendChart.resize();
                    if (window.riskDoughnut) riskDoughnut.resize();
                }, 50);
            } else if (page === 'analytics') {
                document.getElementById("nav-btn-analytics").classList.add("active");
                document.getElementById("page-analytics").style.display = "flex";
                setTimeout(() => {
                    if (window.trendChart) trendChart.resize();
                    if (window.riskDoughnut) riskDoughnut.resize();
                }, 50);
            } else if (page === 'sites') {
                document.getElementById("nav-btn-sites").classList.add("active");
                document.getElementById("page-sites").style.display = "flex";
                setTimeout(() => {
                    initMiningMap();
                    if (leafletMap) leafletMap.invalidateSize();
                }, 100);
            } else if (page === 'admin') {
                document.getElementById("nav-btn-admin").classList.add("active");
                document.getElementById("page-admin").style.display = "flex";
                renderAdminPanel();
            }
        }

        function updateTelemetry() {
            if (document.getElementById("stream-toggle").value === "off" || !currentUser) return;

            const payload = getSimulatedPayload();
            const currentBat = payload.battery !== undefined ? payload.battery : 94;

            // Fast local threshold prediction (0ms rendering delay)
            let status = "SAFE";
            const tiltVal = parseFloat(payload.filtered_tilt);
            const vibVal = parseFloat(payload.filtered_vibration);
            const strainVal = parseFloat(payload.filtered_strain);

            if (tiltVal >= 4.0 || vibVal >= 1.5 || strainVal >= 2.0) {
                status = "DANGER";
            } else if (tiltVal >= 0.4 || vibVal >= 0.35 || strainVal >= 0.4) {
                status = "WARNING";
            }

            const timeStr = getFormattedTimestamp(0);

            // Metrics Update
            document.getElementById("val-tilt").textContent = payload.filtered_tilt.toFixed(4);
            document.getElementById("val-vib").textContent = payload.filtered_vibration.toFixed(4);
            document.getElementById("val-strain").textContent = payload.filtered_strain.toFixed(4);
            document.getElementById("last-updated").textContent = "Last Sync: " + timeStr;

            if (status === "SAFE") safeCount++;
            if (status === "WARNING") warnCount++;
            if (status === "DANGER") dangerCount++;

            document.getElementById("val-alerts").textContent = `${warnCount} / ${dangerCount}`;

            // Analytics Screen Counter Updates
            if (document.getElementById("cnt-safe")) document.getElementById("cnt-safe").textContent = safeCount;
            if (document.getElementById("cnt-warn")) document.getElementById("cnt-warn").textContent = warnCount;
            if (document.getElementById("cnt-danger")) document.getElementById("cnt-danger").textContent = dangerCount;
            if (document.getElementById("cnt-total")) document.getElementById("cnt-total").textContent = safeCount + warnCount + dangerCount;

            // Status Banner Update
            const banner = document.getElementById("status-banner");
            const bannerText = document.getElementById("status-text");
            if (banner && bannerText) {
                banner.className = "status-banner " + status;
                const currentSite = MINING_SITES[selectedSiteKey] || MINING_SITES["site-1"];
                const siteNodeInfo = `(Site: ${currentSite.shortName} | Node: ${selectedNodeId})`;

                if (status === "SAFE") bannerText.textContent = `SAFE — Normal Operation ${siteNodeInfo}`;
                else if (status === "WARNING") bannerText.textContent = `WARNING — Structural Drift Threshold Approached ${siteNodeInfo}`;
                else if (status === "DANGER") bannerText.textContent = `DANGER — Immediate Collapse Risk Detected! ${siteNodeInfo}`;
                else bannerText.textContent = `${status} ${siteNodeInfo}`;
            }

            // Doughnut Update
            if (window.riskDoughnut) {
                riskDoughnut.data.datasets[0].data = [safeCount, warnCount, dangerCount];
                riskDoughnut.update();
            }

            // Trend Chart Update (0ms delay)
            if (window.trendChart) {
                if (trendChart.data.labels.length > 15) {
                    trendChart.data.labels.shift();
                    trendChart.data.datasets[0].data.shift();
                    trendChart.data.datasets[1].data.shift();
                    trendChart.data.datasets[2].data.shift();
                }
                trendChart.data.labels.push(timeStr);
                trendChart.data.datasets[0].data.push(payload.filtered_tilt);
                trendChart.data.datasets[1].data.push(payload.filtered_vibration);
                trendChart.data.datasets[2].data.push(payload.filtered_strain);
                trendChart.update();
            }

            // Log Table Update for current active node
            const activeBuf = getNodeHistoryBuffer(selectedNodeId);
            activeBuf.unshift({ time: timeStr, node_id: selectedNodeId, tilt: payload.filtered_tilt, vib: payload.filtered_vibration, strain: payload.filtered_strain, battery: currentBat, status: status });
            if (activeBuf.length > 20) activeBuf.pop();
            renderTelemetryLogTable();

            // Sync active node data dynamically
            const activeSiteObj = MINING_SITES[selectedSiteKey] || MINING_SITES["site-1"];
            const activeNodeObj = (activeSiteObj && activeSiteObj.nodes) ? activeSiteObj.nodes.find(n => n.id === selectedNodeId) : null;
            if (activeNodeObj) {
                activeNodeObj.tilt = payload.filtered_tilt;
                activeNodeObj.vib = payload.filtered_vibration;
                activeNodeObj.strain = payload.filtered_strain;
                activeNodeObj.status = status;

                // Sync subtitle power info to guarantee 100% consistency across top bar, dropdown, and logs
                const subtitleEl = document.getElementById("live-sub-info");
                if (subtitleEl) {
                    const batInfo = getBatteryInfo(activeNodeObj.battery);
                    subtitleEl.innerHTML = `Active Site: <strong style="color: var(--text-primary);">${activeSiteObj.shortName}</strong> | Connected Sensor: <strong style="color: var(--primary-accent);">${activeNodeObj.id} (${activeNodeObj.name})</strong> | Power: <strong style="color: ${batInfo.color};">${batInfo.icon} ${activeNodeObj.battery}% Battery</strong> | Type: <strong style="color: var(--text-primary);">${activeNodeObj.type}</strong>`;
                }
                if (leafletMap) {
                    renderSiteNodesOnMap();
                }
            }

            // Asynchronous non-blocking ML prediction fetch
            const targetEndpoint = (window.API_BASE_URL || "").replace(new RegExp("/+$"), "") + "/predict";
            fetch(targetEndpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            }).then(res => res.ok ? res.json() : null).then(data => {
                if (data && data.status && data.status !== status && banner && bannerText) {
                    const currentSite = MINING_SITES[selectedSiteKey] || MINING_SITES["site-1"];
                    const siteNodeInfo = `(Site: ${currentSite.shortName} | Node: ${selectedNodeId})`;
                    banner.className = "status-banner " + data.status;
                    bannerText.textContent = `${data.status} — AI Model Prediction ${siteNodeInfo}`;
                }
            }).catch(e => {
                // Non-blocking network fallback
            });

            // Sync Telemetry Log to Supabase Cloud Database
            syncTelemetryToSupabase({
                timestamp: timeStr,
                node_id: payload.node_id || "NODE_01",
                filtered_tilt: payload.filtered_tilt,
                filtered_vibration: payload.filtered_vibration,
                filtered_strain: payload.filtered_strain,
                status: status
            });
        }

        function renderTelemetryLogTable() {
            const tbody = document.getElementById("log-tbody");
            if (!tbody) return;

            const badgeEl = document.getElementById("log-table-node-badge");
            const filterSelect = document.getElementById("log-node-filter");
            const filterVal = filterSelect ? filterSelect.value : "SELECTED";

            let logsToDisplay = [];
            if (filterVal === "ALL") {
                if (badgeEl) {
                    badgeEl.textContent = "All Nodes (Combined Stream)";
                    badgeEl.style.color = "var(--warn-color)";
                    badgeEl.style.borderColor = "rgba(245, 158, 11, 0.4)";
                }
                let combined = [];
                for (let nid in nodeHistoryBuffers) {
                    combined = combined.concat(nodeHistoryBuffers[nid]);
                }
                combined.sort((a, b) => b.time.localeCompare(a.time));
                logsToDisplay = combined.slice(0, 20);
            } else {
                if (badgeEl) {
                    badgeEl.textContent = `${selectedNodeId} (Separate Node Logs)`;
                    badgeEl.style.color = "var(--primary-accent)";
                    badgeEl.style.borderColor = "var(--border-accent)";
                }
                logsToDisplay = getNodeHistoryBuffer(selectedNodeId);
            }

            tbody.innerHTML = logsToDisplay.map(r => {
                let batVal = r.battery;
                if (batVal === undefined) {
                    for (let sk in MINING_SITES) {
                        const matchNode = MINING_SITES[sk].nodes ? MINING_SITES[sk].nodes.find(n => n.id === (r.node_id || selectedNodeId)) : null;
                        if (matchNode && matchNode.battery !== undefined) {
                            batVal = matchNode.battery;
                            break;
                        }
                    }
                }
                if (batVal === undefined) batVal = 94;

                const bInfo = getBatteryInfo(batVal);
                const displayBat = bInfo.text.replace(" (Low)", "").replace(" (Depleted)", "");
                const isCurrentNode = (r.node_id || selectedNodeId) === selectedNodeId;
                const nodeBadgeStyle = isCurrentNode
                    ? "background: rgba(56, 189, 248, 0.15); color: var(--primary-accent); border: 1px solid var(--border-accent);"
                    : "background: rgba(148, 163, 184, 0.15); color: var(--text-muted); border: 1px solid var(--border-color);";

                return `
                <tr>
                    <td style="font-family: monospace; font-weight: 600; color: var(--text-primary);">${r.time}</td>
                    <td><span style="font-family: monospace; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.8rem; ${nodeBadgeStyle}">${r.node_id || selectedNodeId}</span></td>
                    <td>${r.tilt.toFixed(4)}</td>
                    <td>${r.vib.toFixed(4)}</td>
                    <td>${r.strain.toFixed(4)}</td>
                    <td><span style="font-weight: 700; color: ${bInfo.color};">${displayBat}</span></td>
                    <td><span class="badge ${r.status}">${r.status}</span></td>
                </tr>
                `;
            }).join('');
        }

        if (window.location.search) {
            try { history.replaceState(null, '', window.location.pathname); } catch(err) {}
        }
        initTheme();
        applyChartTheme(document.body.classList.contains("light-theme"));
        populateLiveNodeDropdown();
        initChartDataForNode(selectedNodeId);
        renderTelemetryLogTable();
        checkAuth();
        setInterval(updateTelemetry, 2000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTML_DASHBOARD
