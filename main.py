import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import time
import threading
import requests

app = FastAPI(title="Mine Subsidence Early Warning System")

MODEL_FILE = "model.joblib"
model = None

SUPABASE_URL = "https://toabcprwbtaipxwzmdyl.supabase.co"
SUPABASE_KEY = "sb_publishable_DS2T92fPyhKkhGyI41dtzA_qw2_m_3C"

def log_to_supabase_async(payload_dict):
    """Sends telemetry log packet to Supabase Cloud Database in background thread."""
    def send():
        try:
            url = f"{SUPABASE_URL}/rest/v1/telemetry_logs"
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            requests.post(url, json=payload_dict, headers=headers, timeout=3)
        except Exception:
            pass
    threading.Thread(target=send, daemon=True).start()

def get_model():
    global model
    if model is not None:
        return model
    
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            return model
        except Exception as e:
            print(f"Error loading model file: {e}")

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
        return model
    except Exception as e:
        print(f"Fallback model generation failed: {e}")
        return None

@app.on_event("startup")
def startup_event():
    get_model()

class FilteredDataPayload(BaseModel):
    node_id: str
    filtered_tilt: float
    filtered_vibration: float
    filtered_strain: float

@app.post("/predict")
def predict_risk(data: FilteredDataPayload):
    clf = get_model()
    if clf is None:
        raise HTTPException(status_code=503, detail="ML Model unavailable.")
    try:
        features = [[data.filtered_tilt, data.filtered_vibration, data.filtered_strain]]
        prediction = clf.predict(features)[0]
        status_str = str(prediction)

        # Async background sync to Supabase Cloud Database
        log_to_supabase_async({
            "node_id": data.node_id,
            "filtered_tilt": data.filtered_tilt,
            "filtered_vibration": data.filtered_vibration,
            "filtered_strain": data.filtered_strain,
            "status": status_str,
            "timestamp": time.strftime("%H:%M:%S")
        })

        return {"node_id": data.node_id, "status": status_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": get_model() is not None}

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Igniters AI — Subsidence & Hazard Early Warning System</title>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
    <style>
        :root {
            --bg-dark: #090d16;
            --bg-card: #131c2e;
            --bg-card-hover: #1a263e;
            --bg-input: #090d16;
            --border-color: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(56, 189, 248, 0.2);
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
            --header-bg: rgba(19, 28, 46, 0.85);
            --hero-bg: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            --hero-box: #090d16;
            --hero-text: #ffffff;
            --card-shadow: 0 10px 30px rgba(0,0,0,0.3);
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
        header { background: var(--header-bg); backdrop-filter: blur(16px); border-bottom: 1px solid var(--border-color); padding: 1rem 2.5rem; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; transition: background 0.3s; }
        .header-left { display: flex; align-items: center; gap: 2rem; }
        .logo-container { display: flex; align-items: center; gap: 0.8rem; }
        .logo-icon { background: linear-gradient(135deg, #0284c7, #38bdf8); width: 38px; height: 38px; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-size: 1.2rem; box-shadow: 0 0 15px var(--primary-glow); }
        .logo-text { font-size: 1.2rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }
        .logo-text span { color: var(--primary-accent); }

        .nav-links { display: flex; gap: 0.4rem; background: var(--bg-input); padding: 0.3rem; border-radius: 10px; border: 1px solid var(--border-color); }
        .nav-btn { background: transparent; border: none; color: var(--text-muted); padding: 0.5rem 1.1rem; border-radius: 7px; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; }
        .nav-btn:hover { color: var(--text-primary); }
        .nav-btn.active { background: var(--primary-accent); color: #ffffff; font-weight: 700; box-shadow: 0 4px 12px var(--primary-glow); }

        .nav-actions { display: flex; align-items: center; gap: 1rem; }
        .system-pill { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #10b981; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; gap: 0.4rem; }
        .user-badge { background: var(--bg-card); border: 1px solid var(--border-color); padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.85rem; color: var(--text-primary); font-weight: 600; }
        
        .btn-theme-toggle { background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-primary); padding: 0.4rem 0.9rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; cursor: pointer; display: flex; align-items: center; gap: 0.4rem; transition: all 0.2s; }
        .btn-theme-toggle:hover { border-color: var(--primary-accent); color: var(--primary-accent); }

        .btn-logout { background: transparent; color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); padding: 0.4rem 0.9rem; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.85rem; transition: all 0.2s; }
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
        
        .admin-hint { background: rgba(56, 189, 248, 0.08); border: 1px dashed rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 0.75rem; margin-top: 1.2rem; font-size: 0.8rem; color: var(--primary-accent); text-align: center; }

        .auth-msg { margin-top: 1rem; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; display: none; text-align: center; font-weight: 600; }
        .auth-msg.error { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .auth-msg.success { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }

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
        .metric-icon { width: 32px; height: 32px; border-radius: 8px; background: rgba(56, 189, 248, 0.1); color: var(--primary-accent); display: flex; justify-content: center; align-items: center; font-size: 1rem; }
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
                <h2>🚨 IGNITERS AI</h2>
                <p>Enterprise Early Warning System for Mine Subsidence</p>
            </div>
            
            <div class="auth-tabs">
                <div class="auth-tab active" onclick="switchAuthTab('login')">🔑 Sign In</div>
                <div class="auth-tab" onclick="switchAuthTab('register')">📝 Register</div>
            </div>

            <!-- LOGIN FORM -->
            <form id="login-form" onsubmit="handleLogin(event)">
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
            <form id="register-form" style="display: none;" onsubmit="handleRegister(event)">
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

            <div class="admin-hint">
                💡 <strong>Administrator Access:</strong> User: <code>Admin</code> | Pass: <code>godisgreat</code>
            </div>

            <div id="auth-msg" class="auth-msg"></div>
        </div>
    </div>

    <!-- HEADER NAVBAR -->
    <header>
        <div class="header-left">
            <div class="logo-container">
                <div class="logo-icon">🚨</div>
                <div class="logo-text">IGNITERS <span>AI</span></div>
            </div>
            <div class="nav-links">
                <button class="nav-btn active" id="nav-btn-monitoring" onclick="switchPage('monitoring')">🖥️ Live Monitoring</button>
                <button class="nav-btn" id="nav-btn-analytics" onclick="switchPage('analytics')">📊 Telemetry Logs & Risk Analytics</button>
                <button class="nav-btn" id="nav-btn-admin" style="display: none;" onclick="switchPage('admin')">🛡️ Admin Panel</button>
            </div>
        </div>
        <div class="nav-actions">
            <button class="btn-theme-toggle" onclick="toggleTheme()">
                <span id="theme-icon">☀️</span> <span id="theme-text">Light Mode</span>
            </button>
            <div class="system-pill">
                <span style="width: 6px; height: 6px; background: #10b981; border-radius: 50%;"></span> SYSTEM ONLINE
            </div>
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
                <div style="font-size: 0.85rem; color: var(--text-muted); background: var(--bg-input); padding: 0.6rem 1rem; border-radius: 8px; border: 1px solid var(--border-color);">
                    Connected Sensor: <strong style="color: var(--text-primary);">NODE_01</strong> | Model: <strong style="color: var(--primary-accent);">RandomForest (100 Trees)</strong>
                </div>
            </div>

            <!-- HAZARD STATUS BANNER -->
            <div id="status-banner" class="status-banner SAFE">
                <div>STATUS: <span id="status-text">SAFE — Normal Operation</span> (Node: NODE_01)</div>
                <div style="font-size: 0.85rem; font-weight: 500;" id="last-updated">Last Sync: --:--:--</div>
            </div>

            <!-- CONTROLS CARD -->
            <div class="controls-card">
                <div class="controls-left">
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
                        <div class="metric-icon">📐</div>
                    </div>
                    <div class="metric-value" id="val-tilt">0.0245</div>
                    <div class="metric-footer">Deg/m — Structural Gradient</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Filtered Vibration</span>
                        <div class="metric-icon">📳</div>
                    </div>
                    <div class="metric-value" id="val-vib">0.1280</div>
                    <div class="metric-footer">g — Seismic Acceleration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Filtered Strain</span>
                        <div class="metric-icon">⚡</div>
                    </div>
                    <div class="metric-value" id="val-strain">0.0120</div>
                    <div class="metric-footer">mm/m — Micro-Displacement</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Risk Alerts (Buffer)</span>
                        <div class="metric-icon">🚨</div>
                    </div>
                    <div class="metric-value" id="val-alerts">0 / 0</div>
                    <div class="metric-footer">Warnings / Danger Events</div>
                </div>
            </div>

            <!-- FULL WIDTH LIVE TRENDS CHART -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">📈 <span>Real-Time Sensor Telemetry Trends</span></div>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">Last 15 Observations (Live Stream)</span>
                </div>
                <div style="position: relative; height: 380px; width: 100%;">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>

            <!-- QUICK LINK TO SEPARATE ANALYTICS PAGE -->
            <div class="btn-page-link" onclick="switchPage('analytics')">
                <span>📋 Telemetry Buffer Logs & Risk Level Distribution charts have been separated into a dedicated page for clean visibility.</span>
                <span>View Telemetry Logs & Risk Analytics →</span>
            </div>
        </div>

        <!-- SCREEN 2: DEDICATED TELEMETRY LOGS & RISK ANALYTICS -->
        <div id="page-analytics" class="page-container" style="display: none;">
            <div class="hero-banner">
                <div class="hero-title">
                    <h1>📊 Telemetry Buffer Logs & Risk Analytics</h1>
                    <p>Detailed Risk Ratio Distribution, Event Counter Audit, and Historical Sensor Data Table</p>
                </div>
                <button class="nav-btn active" onclick="switchPage('monitoring')">← Back to Live Monitoring</button>
            </div>

            <!-- CUMULATIVE STATS CARDS -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Safe Events</span>
                        <div class="metric-icon" style="color: #10b981;">🟢</div>
                    </div>
                    <div class="metric-value" id="cnt-safe" style="color: #10b981;">0</div>
                    <div class="metric-footer">Normal Operation Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Warnings</span>
                        <div class="metric-icon" style="color: #f59e0b;">🟡</div>
                    </div>
                    <div class="metric-value" id="cnt-warn" style="color: #f59e0b;">0</div>
                    <div class="metric-footer">Drift Threshold Approached</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Danger Alerts</span>
                        <div class="metric-icon" style="color: #ef4444;">🔴</div>
                    </div>
                    <div class="metric-value" id="cnt-danger" style="color: #ef4444;">0</div>
                    <div class="metric-footer">Immediate Hazard Triggers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Total Logged Records</span>
                        <div class="metric-icon">📋</div>
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
                        <div class="panel-title">📊 <span>Risk Level Ratio Distribution</span></div>
                    </div>
                    <div style="position: relative; height: 260px; width: 100%; display: flex; justify-content: center; align-items: center;">
                        <canvas id="riskDoughnut"></canvas>
                    </div>
                </div>

                <!-- FULL TELEMETRY DATA LOG TABLE -->
                <div class="panel-card">
                    <div class="panel-header">
                        <div class="panel-title">📋 <span>Live Sensor Telemetry Log Buffer</span></div>
                        <span style="font-size: 0.78rem; color: var(--text-muted);">Real-Time Audit Log</span>
                    </div>
                    <div class="table-wrapper" style="max-height: 380px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Tilt (deg/m)</th>
                                    <th>Vibration (g)</th>
                                    <th>Strain (mm/m)</th>
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

        <!-- SCREEN 3: DEDICATED ADMIN PANEL -->
        <div id="page-admin" class="page-container" style="display: none;">
            <div class="hero-banner">
                <div class="hero-title">
                    <h1>🛡️ Administrator Authorization & Security Console</h1>
                    <p>Manage User Roles, Authorize Pending Registrations, and Track Active User IP & Browser Sessions</p>
                </div>
                <button class="nav-btn active" onclick="switchPage('monitoring')">← Back to Monitoring</button>
            </div>

            <!-- ADMIN STATS CARDS -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Pending Authorizations</span>
                        <div class="metric-icon" style="color: #f59e0b;">⏳</div>
                    </div>
                    <div class="metric-value" id="adm-cnt-pending" style="color: #f59e0b;">0</div>
                    <div class="metric-footer">Awaiting Admin Approval</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Approved Accounts</span>
                        <div class="metric-icon" style="color: #10b981;">🟢</div>
                    </div>
                    <div class="metric-value" id="adm-cnt-approved" style="color: #10b981;">1</div>
                    <div class="metric-footer">Authorized User Profiles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Active User Sessions</span>
                        <div class="metric-icon">🌐</div>
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
                    <div class="panel-title">👥 <span>All Registered User Accounts</span></div>
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
                    <div class="panel-title">🌐 <span>Active User Sessions & IP Audit Log</span></div>
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
                document.getElementById("theme-icon").textContent = "🌙";
                document.getElementById("theme-text").textContent = "Dark Mode";
            }
        }

        function toggleTheme() {
            const isLight = document.body.classList.toggle("light-theme");
            localStorage.setItem("mine_theme", isLight ? "light" : "dark");
            
            document.getElementById("theme-icon").textContent = isLight ? "🌙" : "☀️";
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
                console.log("⚡ Supabase Cloud Database Connected!");
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

        // Authentication System — Require Admin Authorization for new accounts
        const DEFAULT_USERS = { 
            "Admin": { password: "godisgreat", role: "Administrator", status: "Approved", registeredAt: "2026-09-01 00:00:00" } 
        };

        function getUsers() {
            const saved = localStorage.getItem("mine_users");
            return saved ? JSON.parse(saved) : DEFAULT_USERS;
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
                const u = users[currentUser] || { role: "Operator", status: "Approved" };
                document.getElementById("header-user").textContent = `${currentUser} (${u.role})`;

                // Show Admin Panel tab ONLY for Administrator
                const adminBtn = document.getElementById("nav-btn-admin");
                if (currentUser === "Admin" || u.role === "Administrator") {
                    adminBtn.style.display = "inline-block";
                } else {
                    adminBtn.style.display = "none";
                    if (document.getElementById("page-admin").style.display !== "none") {
                        switchPage('monitoring');
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
            e.preventDefault();
            const u = document.getElementById("login-username").value.trim();
            const p = document.getElementById("login-password").value;
            const users = getUsers();

            if (!users[u] || users[u].password !== p) {
                return showAuthMsg("Invalid username or password.", true);
            }

            // Check Authorization Status
            if (users[u].status !== "Approved" && u !== "Admin") {
                return showAuthMsg("⚠️ Account Pending Authorization! An Administrator must authorize your account in the Admin Panel before you can log in.", true);
            }

            currentUser = u;
            logActiveSession(u, users[u].role);
            showAuthMsg("Login successful! Accessing console...", false);
            setTimeout(checkAuth, 500);
        }

        function handleRegister(e) {
            e.preventDefault();
            const u = document.getElementById("reg-username").value.trim();
            const p = document.getElementById("reg-password").value;
            const r = document.getElementById("reg-role").value;

            if (!u || !p) return showAuthMsg("Username and password are required.", true);
            const users = getUsers();
            if (users[u]) return showAuthMsg("Username already exists.", true);

            users[u] = {
                password: p,
                role: r,
                status: "Pending", // Requires Admin Approval
                registeredAt: new Date().toLocaleString()
            };
            saveUsers(users);
            syncUserToSupabase(u, r, "Pending", users[u].registeredAt);
            showAuthMsg("✅ Account registered! Status: Pending Admin Authorization. Please notify an Administrator to authorize your account.", false);
        }

        function handleLogout() {
            currentUser = null;
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
                            <button onclick="approveUser('${u.username}')" style="background: #10b981; color: #fff; border: none; padding: 0.35rem 0.75rem; border-radius: 6px; font-weight: 700; cursor: pointer; margin-right: 0.4rem;">✅ Authorize</button>
                            <button onclick="rejectUser('${u.username}')" style="background: #ef4444; color: #fff; border: none; padding: 0.35rem 0.75rem; border-radius: 6px; font-weight: 700; cursor: pointer;">❌ Reject</button>
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

        // Telemetry Data Logic
        let historyBuffer = [];
        let safeCount = 0, warnCount = 0, dangerCount = 0;

        function getSimulatedPayload() {
            let tilt, vib, strain;

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
            } else { // Dynamic
                const r = Math.random();
                if (r < 0.75) {
                    tilt = (Math.random() * 0.30 + 0.001).toFixed(4);
                    vib = (Math.random() * 0.30 + 0.01).toFixed(4);
                    strain = (Math.random() * 0.20 + 0.001).toFixed(4);
                } else if (r < 0.90) {
                    tilt = (Math.random() * 3.00 + 0.50).toFixed(4);
                    vib = (Math.random() * 0.85 + 0.35).toFixed(4);
                    strain = (Math.random() * 1.40 + 0.40).toFixed(4);
                } else {
                    tilt = (Math.random() * 8.00 + 4.00).toFixed(4);
                    vib = (Math.random() * 3.50 + 1.50).toFixed(4);
                    strain = (Math.random() * 5.00 + 2.00).toFixed(4);
                }
            }

            return {
                node_id: "NODE_01",
                filtered_tilt: parseFloat(tilt),
                filtered_vibration: parseFloat(vib),
                filtered_strain: parseFloat(strain)
            };
        }

        // Page Switcher Navigation
        function switchPage(page) {
            document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
            document.getElementById("page-monitoring").style.display = "none";
            document.getElementById("page-analytics").style.display = "none";
            document.getElementById("page-admin").style.display = "none";

            if (page === 'monitoring') {
                document.getElementById("nav-btn-monitoring").classList.add("active");
                document.getElementById("page-monitoring").style.display = "flex";
            } else if (page === 'analytics') {
                document.getElementById("nav-btn-analytics").classList.add("active");
                document.getElementById("page-analytics").style.display = "flex";
            } else if (page === 'admin') {
                document.getElementById("nav-btn-admin").classList.add("active");
                document.getElementById("page-admin").style.display = "flex";
                renderAdminPanel();
            }
        }

        async function updateTelemetry() {
            if (document.getElementById("stream-toggle").value === "off" || !currentUser) return;

            const payload = getSimulatedPayload();
            let status = "SAFE";

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    const data = await res.json();
                    status = data.status || "SAFE";
                } else {
                    status = "DISCONNECTED";
                }
            } catch (err) {
                status = "DISCONNECTED";
            }

            const timeStr = new Date().toLocaleTimeString();

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
            banner.className = "status-banner " + status;
            if (status === "SAFE") bannerText.textContent = "SAFE — Normal Operation";
            else if (status === "WARNING") bannerText.textContent = "WARNING — Structural Drift Threshold Approached";
            else if (status === "DANGER") bannerText.textContent = "DANGER — Immediate Collapse Risk Detected!";
            else bannerText.textContent = status;

            // Doughnut Update
            riskDoughnut.data.datasets[0].data = [safeCount, warnCount, dangerCount];
            riskDoughnut.update();

            // Trend Chart Update
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

            // Log Table Update
            historyBuffer.unshift({ time: timeStr, tilt: payload.filtered_tilt, vib: payload.filtered_vibration, strain: payload.filtered_strain, status: status });
            if (historyBuffer.length > 20) historyBuffer.pop();

            // Sync Telemetry Log to Supabase Cloud Database
            syncTelemetryToSupabase({
                timestamp: timeStr,
                node_id: payload.node_id || "NODE_01",
                filtered_tilt: payload.filtered_tilt,
                filtered_vibration: payload.filtered_vibration,
                filtered_strain: payload.filtered_strain,
                status: status
            });

            const tbody = document.getElementById("log-tbody");
            tbody.innerHTML = historyBuffer.map(r => `
                <tr>
                    <td>${r.time}</td>
                    <td>${r.tilt.toFixed(4)}</td>
                    <td>${r.vib.toFixed(4)}</td>
                    <td>${r.strain.toFixed(4)}</td>
                    <td><span class="badge ${r.status}">${r.status}</span></td>
                </tr>
            `).join('');
        }

        initTheme();
        applyChartTheme(document.body.classList.contains("light-theme"));
        checkAuth();
        setInterval(updateTelemetry, 2000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTML_DASHBOARD