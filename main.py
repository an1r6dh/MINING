import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Mine Subsidence Early Warning System")

MODEL_FILE = "model.joblib"
model = None

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
        return {"node_id": data.node_id, "status": str(prediction)}
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
        .logo-container { display: flex; align-items: center; gap: 0.8rem; }
        .logo-icon { background: linear-gradient(135deg, #0284c7, #38bdf8); width: 38px; height: 38px; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-size: 1.2rem; box-shadow: 0 0 15px var(--primary-glow); }
        .logo-text { font-size: 1.2rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }
        .logo-text span { color: var(--primary-accent); }

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

        /* Dashboard Grid Layout */
        .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
        @media (max-width: 1100px) { .dashboard-grid { grid-template-columns: 1fr; } }

        .panel-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.6rem; display: flex; flex-direction: column; gap: 1.2rem; box-shadow: var(--card-shadow); transition: background 0.3s; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; }
        .panel-title { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 0.6rem; }
        .panel-title span { color: var(--primary-accent); }

        /* Manual Input Sliders Box */
        #manual-controls { display: none; background: var(--bg-input); border: 1px solid var(--border-accent); border-radius: 10px; padding: 1rem 1.4rem; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 0.5rem; }
        .slider-group { display: flex; flex-direction: column; gap: 0.3rem; }
        .slider-group label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; display: flex; justify-content: space-between; }
        .slider-group input[type="range"] { width: 100%; accent-color: var(--primary-accent); cursor: pointer; }

        /* Table Styling */
        .table-wrapper { overflow-y: auto; max-height: 360px; border: 1px solid var(--border-color); border-radius: 8px; }
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
                    <input type="text" id="login-username" required placeholder="e.g. Admin">
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
        <div class="logo-container">
            <div class="logo-icon">🚨</div>
            <div class="logo-text">IGNITERS <span>AI</span></div>
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

    <!-- DASHBOARD CONTAINER -->
    <main>
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

        <!-- CHARTS & LOG TABLE GRID -->
        <div class="dashboard-grid">
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">📈 <span>Real-Time Sensor Telemetry Trends</span></div>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">Last 15 Observations</span>
                </div>
                <div style="position: relative; height: 320px; width: 100%;">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>

            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">📊 <span>Risk Level Distribution</span></div>
                </div>
                <div style="position: relative; height: 190px; width: 100%; display: flex; justify-content: center;">
                    <canvas id="riskDoughnut"></canvas>
                </div>
                <div style="margin-top: 1rem; border-top: 1px solid var(--border-color); padding-top: 0.8rem;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">📋 Telemetry Buffer Log</div>
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Tilt</th>
                                    <th>Vib</th>
                                    <th>Strain</th>
                                    <th>Status</th>
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
    </main>

    <!-- FOOTER -->
    <footer>
        <div>© 2026 <strong>Igniters AI</strong> — SIH Project 26025. All Rights Reserved.</div>
        <div>Engineered for Deep Tech Mine Safety & Digital Transformation</div>
    </footer>

    <script>
        // Theme Switcher Logic
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

            // Update Chart.js themes dynamically
            const textColor = isLight ? "#64748b" : "#94a3b8";
            const gridColor = isLight ? "rgba(0, 0, 0, 0.06)" : "rgba(255, 255, 255, 0.05)";
            const legendColor = isLight ? "#0f172a" : "#f8fafc";

            trendChart.options.scales.x.ticks.color = textColor;
            trendChart.options.scales.x.grid.color = gridColor;
            trendChart.options.scales.y.ticks.color = textColor;
            trendChart.options.scales.y.grid.color = gridColor;
            trendChart.options.plugins.legend.labels.color = legendColor;
            trendChart.update();

            riskDoughnut.options.plugins.legend.labels.color = legendColor;
            riskDoughnut.update();
        }

        // Authentication System
        const DEFAULT_USERS = { "Admin": { password: "godisgreat", role: "Administrator" } };
        function getUsers() {
            const saved = localStorage.getItem("mine_users");
            return saved ? JSON.parse(saved) : DEFAULT_USERS;
        }

        let currentUser = localStorage.getItem("mine_current_user");
        let activeSimMode = "dynamic";

        function checkAuth() {
            if (currentUser) {
                document.getElementById("auth-screen").style.display = "none";
                const users = getUsers();
                const u = users[currentUser] || { role: "Operator" };
                document.getElementById("header-user").textContent = `${currentUser} (${u.role})`;
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

            if (users[u] && users[u].password === p) {
                currentUser = u;
                localStorage.setItem("mine_current_user", u);
                showAuthMsg("Login successful! Redirecting to console...", false);
                setTimeout(checkAuth, 500);
            } else {
                showAuthMsg("Invalid username or password.", true);
            }
        }

        function handleRegister(e) {
            e.preventDefault();
            const u = document.getElementById("reg-username").value.trim();
            const p = document.getElementById("reg-password").value;
            const r = document.getElementById("reg-role").value;

            if (!u || !p) return showAuthMsg("Username and password are required.", true);
            const users = getUsers();
            if (users[u]) return showAuthMsg("Username already exists.", true);

            users[u] = { password: p, role: r };
            localStorage.setItem("mine_users", JSON.stringify(users));
            showAuthMsg("Account created! Switch to Sign In tab to log in.", false);
        }

        function handleLogout() {
            currentUser = null;
            localStorage.removeItem("mine_current_user");
            checkAuth();
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
        checkAuth();
        setInterval(updateTelemetry, 2000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTML_DASHBOARD