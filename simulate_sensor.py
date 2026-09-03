import random
import sys
import time
import requests

URL = "http://127.0.0.1:8000/predict"

print("[INFO] Starting Mine Sensor Telemetry Simulator...")
print("Press CTRL+C at any time to stop the sensor stream.\n")

try:
    while True:
        # Determine simulated hazard scenario dynamically (75% SAFE, 15% WARNING, 10% DANGER)
        chance = random.random()
        if chance < 0.75:
            # SAFE range
            tilt = round(random.uniform(0.001, 0.30), 4)
            vib = round(random.uniform(0.01, 0.30), 4)
            strain = round(random.uniform(0.001, 0.20), 4)
        elif chance < 0.90:
            # WARNING range
            tilt = round(random.uniform(0.50, 3.50), 4)
            vib = round(random.uniform(0.35, 1.20), 4)
            strain = round(random.uniform(0.40, 1.80), 4)
        else:
            # DANGER range
            tilt = round(random.uniform(4.00, 12.00), 4)
            vib = round(random.uniform(1.50, 5.00), 4)
            strain = round(random.uniform(2.00, 7.00), 4)

        payload = {
            "node_id": "NODE_01",
            "filtered_tilt": tilt,
            "filtered_vibration": vib,
            "filtered_strain": strain,
        }

        try:
            response = requests.post(URL, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "UNKNOWN")
                print(
                    f"[SENT] Tilt: {tilt:.4f} | Vibration: {vib:.4f} | Strain: {strain:.4f}"
                )
                print(
                    f"[RESPONSE] Node: {result['node_id']} -> Risk Status: {status}\n"
                )
            else:
                print(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("[WARNING] Could not connect to FastAPI server. Make sure uvicorn is running on http://127.0.0.1:8000!")
        except requests.exceptions.Timeout:
            print("[WARNING] Request to FastAPI server timed out.")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")

        time.sleep(2)
except KeyboardInterrupt:
    print("\n[STOP] Sensor telemetry stream stopped by user. Exiting cleanly.")
    sys.exit(0)