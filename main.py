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
    <title>Mine Subsidence Early Warning System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --safe-color: #22c55e;
            --warn-color: #eab308;
            --danger-color: #ef4444;
            --border-color: #334155;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); min-height: 100vh; display: flex; flex-direction: column; }

        /* Navigation Header */
        header { background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border-bottom: 1px solid var(--border-color); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
        .logo-title { display: flex; align-items: center; gap: 0.75rem; font-size: 1.25rem; font-weight: 700; color: #fff; }
        .user-profile { display: flex; align-items: center; gap: 1rem; }
        .user-badge { background: #334155; padding: 0.4rem 0.8rem; borderRadius: 20px; font-size: 0.85rem; color: var(--accent-blue); font-weight: 600; }
        .btn-logout { background: #ef4444; color: white; border: none; padding: 0.4rem 0.9rem; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; transition: 0.2s; }
        .btn-logout:hover { background: #dc2626; }

        /* Auth Screen Overlay */
        #auth-screen { position: fixed; inset: 0; background: var(--bg-color); z-index: 200; display: flex; justify-content: center; align-items: center; padding: 1.5rem; }
        .auth-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 2.5rem; width: 100%; max-width: 440px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }
        .auth-tabs { display: flex; border-bottom: 1px solid var(--border-color); margin-bottom: 1.5rem; }
        .auth-tab { flex: 1; padding: 0.75rem; text-align: center; cursor: pointer; font-weight: 600; color: var(--text-secondary); border-bottom: 2px solid transparent; transition: 0.2s; }
        .auth-tab.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue); }
        .form-group { margin-bottom: 1.2rem; }
        .form-group label { display: block; margin-bottom: 0.4rem; font-size: 0.85rem; color: var(--text-secondary); }
        .form-group input, .form-group select { width: 100%; padding: 0.75rem; background: #0f172a; border: 1px solid var(--border-color); border-radius: 6px; color: #fff; font-size: 0.95rem; }
        .form-group input:focus { outline: none; border-color: var(--accent-blue); }
        .btn-submit { width: 100%; padding: 0.85rem; background: var(--accent-blue); color: #0f172a; border: none; border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 1rem; transition: 0.2s; margin-top: 0.5rem; }
        .btn-submit:hover { background: #0284c7; color: #fff; }
        .auth-msg { margin-top: 1rem; padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; display: none; text-align: center; }
        .auth-msg.error { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
        .auth-msg.success { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid #22c55e; }

        /* Main Dashboard Content */
        main { padding: 2rem; max-width: 1400px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; gap: 1.5rem; flex: 1; }

        /* Status Alert Banner */
        .status-banner { padding: 1.25rem 1.5rem; border-radius: 10px; font-size: 1.2rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between; border: 1px solid transparent; transition: 0.3s; }
        .status-banner.SAFE { background: rgba(34, 197, 94, 0.15); color: #4ade80; border-color: var(--safe-color); }
        .status-banner.WARNING { background: rgba(234, 179, 8, 0.15); color: #fde047; border-color: var(--warn-color); }
        .status-banner.DANGER { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border-color: var(--danger-color); }
        .status-banner.DISCONNECTED { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border-color: var(--text-secondary); }

        /* Metric Grid */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem; }
        .metric-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.5rem; }
        .metric-label { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
        .metric-unit { font-size: 0.85rem; color: var(--accent-blue); font-weight: 600; }

        /* Controls Section */
        .controls-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; padding: 1.25rem; display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; }
        .control-group { display: flex; flex-direction: column; gap: 0.4rem; }
        .control-group label { font-size: 0.85rem; color: var(--text-secondary); font-weight: 600; }
        .control-group select, .control-group input[type="range"] { background: #0f172a; border: 1px solid var(--border-color); color: #fff; padding: 0.5rem 0.8rem; border-radius: 6px; }

        /* Visual Trends & Log Grid */
        .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
        @media (max-width: 1024px) { .dashboard-grid { grid-template-columns: 1fr; } }
        
        .panel-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 10px; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
        .panel-title { font-size: 1.1rem; font-weight: 700; color: #fff; }

        /* Telemetry Table */
        .table-wrapper { overflow-y: auto; max-height: 340px; border: 1px solid var(--border-color); border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; }
        th { background: #0f172a; color: var(--text-secondary); padding: 0.75rem 1rem; position: sticky; top: 0; }
        td { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); }
        tr:nth-child(even) { background: rgba(15, 23, 42, 0.4); }
        
        .badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.75rem; display: inline-block; }
        .badge.SAFE { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        .badge.WARNING { background: rgba(234, 179, 8, 0.2); color: #fde047; }
        .badge.DANGER { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
    </style>
</head>
<body>

    <!-- AUTHENTICATION OVERLAY -->
    <div id="auth-screen">
        <div class="auth-card">
            <h2 style="text-align: center; margin-bottom: 0.5rem; color: #fff;">🚨 Mine Hazard Monitoring</h2>
            <p style="text-align: center; color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1.5rem;">SIH Project 26025 Early Warning Dashboard</p>
            
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
                <button type="submit" class="btn-submit">Sign In</button>
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

            <div id="auth-msg" class="auth-msg"></div>
        </div>
    </div>

    <!-- MAIN APP HEADER -->
    <header>
        <div class="logo-title">
            <span>🚨</span> Mine Subsidence Early Warning System
        </div>
        <div class="user-profile">
            <span class="user-badge" id="header-user">Admin (Administrator)</span>
            <button class="btn-logout" onclick="handleLogout()">Logout</button>
        </div>
    </header>

    <!-- DASHBOARD CONTAINER -->
    <main>
        <!-- STATUS BANNER -->
        <div id="status-banner" class="status-banner SAFE">
            <div>STATUS: <span id="status-text">SAFE — Normal Operation</span> (Node: NODE_01)</div>
            <div style="font-size: 0.85rem; font-weight: 400;" id="last-updated">Updated: --:--:--</div>
        </div>

        <!-- CONTROLS -->
        <div class="controls-card">
            <div class="control-group">
                <label>Simulation Mode</label>
                <select id="sim-mode">
                    <option value="dynamic">Dynamic (Random Hazards)</option>
                    <option value="safe">Normal (Safe)</option>
                    <option value="warning">Drift (Warning)</option>
                    <option value="danger">Hazard (Danger)</option>
                </select>
            </div>
            <div class="control-group">
                <label>Live Sensor Polling</label>
                <select id="stream-toggle">
                    <option value="on">Enabled (Every 2s)</option>
                    <option value="off">Paused</option>
                </select>
            </div>
        </div>

        <!-- METRICS CARDS -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Filtered Tilt</div>
                <div class="metric-value" id="val-tilt">0.0245</div>
                <div class="metric-unit">Deg/m</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Filtered Vibration</div>
                <div class="metric-value" id="val-vib">0.1280</div>
                <div class="metric-unit">g</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Filtered Strain</div>
                <div class="metric-value" id="val-strain">0.0120</div>
                <div class="metric-unit">mm/m</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Alert Summary (Buffer)</div>
                <div class="metric-value" id="val-alerts">0 / 0</div>
                <div class="metric-unit">Warn / Danger</div>
            </div>
        </div>

        <!-- CHARTS & LOG TABLE -->
        <div class="dashboard-grid">
            <div class="panel-card">
                <div class="panel-title">📈 Live Sensor Telemetry Trends</div>
                <div style="position: relative; height: 320px; width: 100%;">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>
            <div class="panel-card">
                <div class="panel-title">📋 Telemetry Log Buffer</div>
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
                            <!-- Rows inserted dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Authentication System Storage
        const DEFAULT_USERS = { "Admin": { password: "godisgreat", role: "Administrator" } };
        function getUsers() {
            const saved = localStorage.getItem("mine_users");
            return saved ? JSON.parse(saved) : DEFAULT_USERS;
        }

        let currentUser = localStorage.getItem("mine_current_user");

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
                showAuthMsg("Login successful! Loading dashboard...", false);
                setTimeout(checkAuth, 600);
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

        // Chart Initialization
        const ctx = document.getElementById('trendChart').getContext('2d');
        const trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Tilt (deg/m)', data: [], borderColor: '#38bdf8', tension: 0.3, fill: false },
                    { label: 'Vibration (g)', data: [], borderColor: '#eab308', tension: 0.3, fill: false },
                    { label: 'Strain (mm/m)', data: [], borderColor: '#ef4444', tension: 0.3, fill: false }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });

        // Telemetry Data Simulation & Fetch Loop
        let historyBuffer = [];
        let warnCount = 0;
        let dangerCount = 0;

        function getSimulatedPayload() {
            const mode = document.getElementById("sim-mode").value;
            let tilt, vib, strain;

            if (mode === "safe") {
                tilt = (Math.random() * 0.30 + 0.001).toFixed(4);
                vib = (Math.random() * 0.30 + 0.01).toFixed(4);
                strain = (Math.random() * 0.20 + 0.001).toFixed(4);
            } else if (mode === "warning") {
                tilt = (Math.random() * 3.00 + 0.50).toFixed(4);
                vib = (Math.random() * 0.85 + 0.35).toFixed(4);
                strain = (Math.random() * 1.40 + 0.40).toFixed(4);
            } else if (mode === "danger") {
                tilt = (Math.random() * 8.00 + 4.00).toFixed(4);
                vib = (Math.random() * 3.50 + 1.50).toFixed(4);
                strain = (Math.random() * 5.00 + 2.00).toFixed(4);
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

            // Update Metrics Cards
            document.getElementById("val-tilt").textContent = payload.filtered_tilt.toFixed(4);
            document.getElementById("val-vib").textContent = payload.filtered_vibration.toFixed(4);
            document.getElementById("val-strain").textContent = payload.filtered_strain.toFixed(4);
            document.getElementById("last-updated").textContent = "Updated: " + timeStr;

            if (status === "WARNING") warnCount++;
            if (status === "DANGER") dangerCount++;
            document.getElementById("val-alerts").textContent = `${warnCount} / ${dangerCount}`;

            // Update Status Banner
            const banner = document.getElementById("status-banner");
            const bannerText = document.getElementById("status-text");
            banner.className = "status-banner " + status;
            if (status === "SAFE") bannerText.textContent = "SAFE — Normal Operation";
            else if (status === "WARNING") bannerText.textContent = "WARNING — Structural Drift Threshold Approached";
            else if (status === "DANGER") bannerText.textContent = "DANGER — Immediate Collapse Risk Detected!";
            else bannerText.textContent = status;

            // Update Chart
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

            // Update Log Table
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

        checkAuth();
        setInterval(updateTelemetry, 2000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTML_DASHBOARD