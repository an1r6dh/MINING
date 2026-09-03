import json
import os
import random
import time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Mine Safety Telemetry Dashboard",
    page_icon="🚨",
    layout="wide",
)

USER_DB_FILE = "users.json"
API_URL = "http://127.0.0.1:8000/predict"

def load_users():
    """Loads user credentials database with default secret Admin access."""
    default_users = {
        "Admin": {"password": "godisgreat", "role": "Administrator"}
    }
    if not os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "w") as f:
                json.dump(default_users, f, indent=4)
        except Exception:
            pass
        return default_users
    try:
        with open(USER_DB_FILE, "r") as f:
            users = json.load(f)
            # Ensure secret Admin access is always present
            if "Admin" not in users or users["Admin"].get("password") != "godisgreat":
                users["Admin"] = {"password": "godisgreat", "role": "Administrator"}
                with open(USER_DB_FILE, "w") as wf:
                    json.dump(users, wf, indent=4)
            return users
    except Exception:
        return default_users

def save_user(username, password, role="Operator"):
    """Saves newly registered user into persistent storage."""
    users = load_users()
    users[username] = {"password": password, "role": role}
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Initialize Session States
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------
# 🔒 AUTHENTICATION SCREEN (SIGN IN / REGISTER)
# ----------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center;'>🚨 Mine Subsidence Early Warning System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Please sign in or create an account to access the monitoring dashboard.</p>", unsafe_allow_html=True)
    st.divider()

    col_center, _, _ = st.columns([2, 1, 1]) if False else (st.container(), None, None)
    
    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])

    with auth_col2:
        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Register"])

        # LOGIN TAB
        with tab_login:
            st.subheader("Sign In to Account")
            login_user = st.text_input("Username", key="login_username")
            login_pass = st.text_input("Password", type="password", key="login_password")

            if st.button("Sign In", use_container_width=True, type="primary"):
                users = load_users()
                if login_user in users and users[login_user]["password"] == login_pass:
                    st.session_state.authenticated = True
                    st.session_state.username = login_user
                    st.session_state.role = users[login_user].get("role", "User")
                    st.success(f"Welcome back, {login_user}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

        # REGISTER TAB
        with tab_register:
            st.subheader("Create New Account")
            reg_user = st.text_input("New Username", key="reg_username")
            reg_pass = st.text_input("New Password", type="password", key="reg_password")
            reg_pass_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
            reg_role = st.selectbox("Role", ["Operator", "Safety Engineer", "Inspector"])

            if st.button("Register Account", use_container_width=True):
                users = load_users()
                if not reg_user or not reg_pass:
                    st.error("Username and password fields cannot be empty.")
                elif reg_user in users:
                    st.error("Username already exists! Please choose a different username.")
                elif reg_pass != reg_pass_confirm:
                    st.error("Passwords do not match.")
                else:
                    save_user(reg_user, reg_pass, reg_role)
                    st.success("Account created successfully! You can now switch to the 'Sign In' tab.")

    st.stop()

# ----------------------------------------------------
# 🚨 AUTHENTICATED DASHBOARD VIEW
# ----------------------------------------------------
st.sidebar.markdown(f"### 👤 Logged in as: **{st.session_state.username}**")
st.sidebar.markdown(f"🏷️ Role: `{st.session_state.role}`")

if st.sidebar.button("🚪 Logout", type="secondary"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()

# Sidebar Theme Control
st.sidebar.header("🎨 Appearance")
theme_mode = st.sidebar.radio("Dashboard Theme", ["🌙 Dark Mode", "☀️ Light Mode"], index=0)

if theme_mode == "☀️ Light Mode":
    st.markdown("""
        <style>
        .stApp { background-color: #f1f5f9; color: #0f172a; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #cbd5e1; }
        [data-testid="stHeader"] { background-color: rgba(255, 255, 255, 0.8); }
        .stMetric { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        div[data-testid="stMetricValue"] { color: #0f172a !important; }
        h1, h2, h3, p, label { color: #0f172a !important; }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.divider()

# Sidebar Controls
st.sidebar.header("🕹️ Controls & Simulation")
sim_mode = st.sidebar.selectbox(
    "Simulation Mode",
    ["Dynamic (Random Hazards)", "Normal (Safe)", "Drift (Warning)", "Hazard (Danger)", "Manual Input"]
)

if sim_mode == "Manual Input":
    manual_tilt = st.sidebar.slider("Filtered Tilt (deg/m)", 0.0, 15.0, 0.05, step=0.01)
    manual_vib = st.sidebar.slider("Filtered Vibration (g)", 0.0, 6.0, 0.10, step=0.01)
    manual_strain = st.sidebar.slider("Filtered Strain (mm/m)", 0.0, 8.0, 0.02, step=0.01)
    auto_stream = False
    send_manual = st.sidebar.button("Send Telemetry Reading")
else:
    auto_stream = st.sidebar.checkbox("Enable Live Sensor Stream", value=True)
    refresh_rate = st.sidebar.slider("Polling Frequency (seconds)", 1, 5, 2)
    send_manual = False

if st.sidebar.button("Clear Log History"):
    st.session_state.history = []
    st.rerun()

def get_telemetry_payload(mode):
    if mode == "Normal (Safe)":
        tilt = round(random.uniform(0.001, 0.30), 4)
        vib = round(random.uniform(0.01, 0.30), 4)
        strain = round(random.uniform(0.001, 0.20), 4)
    elif mode == "Drift (Warning)":
        tilt = round(random.uniform(0.50, 3.50), 4)
        vib = round(random.uniform(0.35, 1.20), 4)
        strain = round(random.uniform(0.40, 1.80), 4)
    elif mode == "Hazard (Danger)":
        tilt = round(random.uniform(4.00, 12.00), 4)
        vib = round(random.uniform(1.50, 5.00), 4)
        strain = round(random.uniform(2.00, 7.00), 4)
    elif mode == "Manual Input":
        tilt, vib, strain = manual_tilt, manual_vib, manual_strain
    else: # Dynamic
        chance = random.random()
        if chance < 0.75:
            tilt = round(random.uniform(0.001, 0.30), 4)
            vib = round(random.uniform(0.01, 0.30), 4)
            strain = round(random.uniform(0.001, 0.20), 4)
        elif chance < 0.90:
            tilt = round(random.uniform(0.50, 3.50), 4)
            vib = round(random.uniform(0.35, 1.20), 4)
            strain = round(random.uniform(0.40, 1.80), 4)
        else:
            tilt = round(random.uniform(4.00, 12.00), 4)
            vib = round(random.uniform(1.50, 5.00), 4)
            strain = round(random.uniform(2.00, 7.00), 4)

    return {
        "node_id": "NODE_01",
        "filtered_tilt": tilt,
        "filtered_vibration": vib,
        "filtered_strain": strain,
    }

def fetch_prediction(payload):
    try:
        res = requests.post(API_URL, json=payload, timeout=2)
        if res.status_code == 200:
            result = res.json()
            return result.get("status", "UNKNOWN")
        else:
            return f"HTTP {res.status_code}"
    except Exception:
        return "BACKEND DISCONNECTED"

# Process stream or manual trigger
should_update = auto_stream or send_manual

if should_update:
    payload = get_telemetry_payload(sim_mode)
    status = fetch_prediction(payload)

    record = {
        "Time": time.strftime("%H:%M:%S"),
        "Tilt": payload["filtered_tilt"],
        "Vibration": payload["filtered_vibration"],
        "Strain": payload["filtered_strain"],
        "Status": status,
    }
    st.session_state.history.append(record)
    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

# Display Dashboard UI
df = pd.DataFrame(st.session_state.history)

if not df.empty:
    latest = df.iloc[-1]
    status = latest["Status"]
    tilt_val, vib_val, strain_val = latest["Tilt"], latest["Vibration"], latest["Strain"]
else:
    status = "NO DATA"
    tilt_val, vib_val, strain_val = 0.0, 0.0, 0.0

# Live Risk Alert Banner
if status == "SAFE":
    st.success("### STATUS: SAFE — Normal Operation (Node: NODE_01)")
elif status == "WARNING":
    st.warning("### STATUS: WARNING — Structural Drift Threshold Approached (Node: NODE_01)")
elif status == "DANGER":
    st.error("### STATUS: DANGER — Immediate Collapse Risk Detected! (Node: NODE_01)")
else:
    st.info(f"### STATUS: {status}")

# Metrics Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="Filtered Tilt", value=f"{tilt_val:.4f}", delta="Deg/m")
c2.metric(label="Filtered Vibration", value=f"{vib_val:.4f}", delta="g")
c3.metric(label="Filtered Strain", value=f"{strain_val:.4f}", delta="mm/m")

if not df.empty:
    warn_ct = len(df[df["Status"] == "WARNING"])
    dang_ct = len(df[df["Status"] == "DANGER"])
    c4.metric(label="Alert Summary (Buffer)", value=f"{len(df)} total", delta=f"{warn_ct} Warn / {dang_ct} Danger")
else:
    c4.metric(label="Alert Summary", value="0 total")

st.divider()

# Telemetry History Chart and Table
if not df.empty:
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.subheader("📈 Live Sensor Trends")
        st.line_chart(df.set_index("Time")[["Tilt", "Vibration", "Strain"]])

    with col_table:
        st.subheader("📋 Telemetry Log")
        st.dataframe(df.iloc[::-1], use_container_width=True)

if auto_stream:
    time.sleep(refresh_rate)
    st.rerun()