import argparse
import os
import random
import sys
import time
import requests

# Ultrasonic sensor baseline configuration
BASELINE_DISTANCE_MM = 1000.0  # Ultrasonic initial distance to wall in mm
REFERENCE_LENGTH_METERS = 1.0   # Baseline length L0 in meters

parser = argparse.ArgumentParser(description="Mine Sensor Telemetry Stream Simulator (Ultrasonic Sensor Enabled)")
parser.add_argument("--url", type=str, default=None, help="Target API /predict URL (e.g. Cloudflare Tunnel URL)")
args, _ = parser.parse_known_args()

URL = args.url or os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000/predict"

print(f"[INFO] Starting Ultrasonic Mine Sensor Simulator -> Target URL: {URL}")
print(f"[INFO] Ultrasonic Baseline Distance: {BASELINE_DISTANCE_MM} mm | Reference Length L0: {REFERENCE_LENGTH_METERS} m")
print("Press CTRL+C at any time to stop the sensor stream.\n")

try:
    while True:
        # Determine simulated hazard scenario dynamically (75% SAFE, 15% WARNING, 10% DANGER)
        chance = random.random()
        if chance < 0.75:
            # SAFE range: displacement 0.001mm to 0.20mm
            tilt = round(random.uniform(0.001, 0.30), 4)
            vib = round(random.uniform(0.01, 0.30), 4)
            displacement_mm = round(random.uniform(0.001, 0.20), 4)
        elif chance < 0.90:
            # WARNING range: displacement 0.40mm to 1.80mm
            tilt = round(random.uniform(0.50, 3.50), 4)
            vib = round(random.uniform(0.35, 1.20), 4)
            displacement_mm = round(random.uniform(0.40, 1.80), 4)
        else:
            # DANGER range: displacement 2.00mm to 7.00mm
            tilt = round(random.uniform(4.00, 12.00), 4)
            vib = round(random.uniform(1.50, 5.00), 4)
            displacement_mm = round(random.uniform(2.00, 7.00), 4)

        # Calculate raw Ultrasonic reading and derived Filtered Strain (mm/m)
        ultrasonic_distance_mm = round(BASELINE_DISTANCE_MM - displacement_mm, 4)
        strain = round(displacement_mm / REFERENCE_LENGTH_METERS, 4)

        battery = round(random.uniform(90.0, 99.0), 1)
        node_id = f"NODE_{random.randint(1, 20):02d}"

        payload = {
            "node_id": node_id,
            "filtered_tilt": tilt,
            "filtered_vibration": vib,
            "filtered_strain": strain,
            "battery": battery,
        }

        try:
            response = requests.post(URL, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "UNKNOWN")
                print(
                    f"[SENT] Node: {node_id} | Tilt: {tilt:.4f}° | Vib: {vib:.4f}g | Ultrasonic: {ultrasonic_distance_mm:.2f}mm (Displacement: Δ{displacement_mm:.4f}mm -> Strain: {strain:.4f}mm/m)"
                )
                print(
                    f"[RESPONSE] Node: {result.get('node_id', node_id)} -> AI Risk Status: {status}\n"
                )
            else:
                print(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"[WARNING] Could not connect to target server ({URL}). Make sure backend is active!")
        except requests.exceptions.Timeout:
            print("[WARNING] Request to FastAPI server timed out.")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")

        time.sleep(2)
except KeyboardInterrupt:
    print("\n[STOP] Sensor telemetry stream stopped by user. Exiting cleanly.")
    sys.exit(0)