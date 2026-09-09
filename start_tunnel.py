import os
import sys
import shutil
import subprocess
import re
import time
import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LOCAL_PORT = 8000
LOCAL_URL = f"http://localhost:{LOCAL_PORT}"

def check_cloudflared():
    path = shutil.which("cloudflared")
    if path:
        return path
    
    # Common Windows winget install paths
    possible_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"),
        r"C:\Program Files\cloudflared\cloudflared.exe",
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def check_backend_running():
    try:
        r = requests.get(f"{LOCAL_URL}/health", timeout=2)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False

def main():
    print("============================================================")
    print(" [START] Mine Subsidence EWS - Cloudflare Tunnel Helper Script")
    print("============================================================\n")

    cf_executable = check_cloudflared()
    if not cf_executable:
        print("[ERROR] 'cloudflared' CLI tool not found on system PATH.\n")
        print("Please install Cloudflare Tunnel using WinGet by running:")
        print("   winget install Cloudflare.cloudflared\n")
        print("After installation, restart your terminal and rerun this script.")
        return

    print(f"[OK] Found cloudflared at: {cf_executable}")

    if not check_backend_running():
        print(f"[WARNING] FastAPI backend is NOT responding on {LOCAL_URL}.")
        print("   Attempting to start Uvicorn backend automatically...")
        try:
            subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(LOCAL_PORT)])
            time.sleep(3)
        except Exception as e:
            print(f"   [ERROR] Could not start backend: {e}")
            print(f"   Please start backend manually: uvicorn main:app --port {LOCAL_PORT}")
    else:
        print(f"[OK] Local FastAPI backend is active on {LOCAL_URL}")

    print("\n------------------------------------------------------------")
    print("[START] Starting Cloudflare Tunnel to expose local GPU backend...")
    print("------------------------------------------------------------\n")

    cmd = [cf_executable, "tunnel", "--url", LOCAL_URL]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        tunnel_url = None
        url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

        for line in process.stdout:
            print(line, end="")
            match = url_regex.search(line)
            if match and not tunnel_url:
                tunnel_url = match.group(0)
                print("\n" + "=" * 64)
                print(" [SUCCESS] CLOUDFLARE TUNNEL LIVE!")
                print("=" * 64)
                print(f" Public Backend Base URL   : {tunnel_url}")
                print(f" Telemetry API Route       : {tunnel_url}/api/telemetry")
                print(f" Prediction Route          : {tunnel_url}/predict")
                print(f" Health Check Route        : {tunnel_url}/health")
                print("=" * 64)
                print("\n[TEST] Test Your Tunnel Endpoint (PowerShell):")
                print(f'   Invoke-RestMethod -Uri "{tunnel_url}/api/telemetry" -Method Post -ContentType "application/json" -Body \'{{\"node_id\":\"NODE-01\",\"filtered_tilt\":0.002,\"filtered_vibration\":0.001,\"filtered_strain\":0.003}}\'')
                print("\n[INFO] Integration Instructions:")
                print(f" 1. Vercel Project Settings -> Environment Variables:")
                print(f"    Set API_URL = {tunnel_url}")
                print(f" 2. Streamlit Dashboard:")
                print(f'    $env:API_URL="{tunnel_url}/api/telemetry"; streamlit run dashboard.py')
                print(f" 3. IoT Sensor Simulator:")
                print(f"    python simulate_sensor.py --url {tunnel_url}/api/telemetry\n")
                print("Press CTRL+C at any time to terminate the tunnel.\n")

        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Cloudflare Tunnel. Exiting cleanly.")
        sys.exit(0)

if __name__ == "__main__":
    main()
