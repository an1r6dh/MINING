"""
====================================================================================================
PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
MODULE : USB Serial Hardware Bridge to Web Dashboard & Email Alert System
TARGET : Listens to ESP32-S3 Central Master Hub (central_hub_esp32s3_5.ino) @ 115200 Baud
====================================================================================================
"""

import sys
import os
import re
import time
import json
import argparse
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional PySerial import
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# Regex matching ESP32-S3 output format:
# "[NODE 1 KALMAN FILTERED] Tilt: +00.00 deg | Vib: 00.00 g | Disp: 000.0 mm"
PATTERN_TELEMETRY = re.compile(
    r"\[NODE\s+(\d+)\s+([^\]]+)\]\s+Tilt:\s*([+-]?\d+(?:\.\d+)?)\s*deg\s*\|\s*Vib:\s*([+-]?\d+(?:\.\d+)?)\s*g\s*\|\s*Disp:\s*([+-]?\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE
)

# Regex matching Alert format from central_hub_esp32s3_5.ino:
# "[LOCAL ALERT] Node 1 breached safety limit! Tilt: 5.2 deg, Disp: 16.2 mm"
# Also supports optional Vib parameter if present
PATTERN_ALERT = re.compile(
    r"\[LOCAL ALERT\]\s+Node\s+(\d+)\s+breached safety limit!\s+Tilt:\s*([+-]?\d+(?:\.\d+)?)\s*deg(?:,\s*Vib:\s*([+-]?\d+(?:\.\d+)?)\s*g)?,\s*Disp:\s*([+-]?\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE
)

def post_telemetry_to_dashboard(payload, target_url="http://127.0.0.1:8000/api/hardware_telemetry"):
    """Posts live parsed hardware readings to the dashboard backend."""
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False

def list_available_ports():
    """Returns a list of all detected COM ports."""
    if not HAS_SERIAL:
        return []
    return [port.device for port in serial.tools.list_ports.comports()]

def run_hardware_listener(port, baudrate=115200, target_url="http://127.0.0.1:8000/api/hardware_telemetry"):
    """Reads live serial data from the physical ESP32-S3 Central Hub."""
    if not HAS_SERIAL:
        print("[ERROR] pyserial is not installed. Run: pip install pyserial")
        return

    print(f"\n=======================================================")
    print(f"[BRIDGE] OPENING USB SERIAL CONNECTION TO ESP32-S3")
    print(f"Port      : {port}")
    print(f"Baud Rate : {baudrate}")
    print(f"Backend   : {target_url}")
    print(f"=======================================================")

    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=1.0)
        print(f"[SUCCESS] Connected to {port}! Listening for live ESP-NOW TDMA frames...\n")
    except Exception as e:
        print(f"[ERROR] Could not open serial port {port}: {e}")
        return

    try:
        while True:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                print(f"[ESP32-S3 RX] {raw_line}")

                # 1. Check for standard telemetry line
                match_tel = PATTERN_TELEMETRY.search(raw_line)
                if match_tel:
                    node_id = int(match_tel.group(1))
                    filter_type = match_tel.group(2).strip()
                    tilt = float(match_tel.group(3))
                    vib = float(match_tel.group(4))
                    disp = float(match_tel.group(5))

                    is_danger = (tilt >= 5.0 or vib >= 0.8 or disp >= 15.0)
                    is_warning = (tilt >= 2.5 or vib >= 0.4 or disp >= 8.0)
                    status = "DANGER" if is_danger else ("WARNING" if is_warning else "SAFE")

                    payload = {
                        "source": "ESP32_S3_HARDWARE",
                        "node_id": f"NODE_0{node_id}",
                        "filter_mode": filter_type,
                        "tilt": tilt,
                        "vibration": vib,
                        "displacement": disp,
                        "status": status,
                        "timestamp": time.strftime("%H:%M:%S")
                    }

                    posted = post_telemetry_to_dashboard(payload, target_url)
                    indicator = "[WEB SYNC OK]" if posted else "[LOCAL ONLY]"
                    print(f"       --> Parsed Node {node_id}: Tilt={tilt:+.2f}deg | Vib={vib:.2f}g | Disp={disp:.1f}mm | {status} {indicator}")

                # 2. Check for local alert line
                match_alt = PATTERN_ALERT.search(raw_line)
                if match_alt:
                    node_id = int(match_alt.group(1))
                    tilt = float(match_alt.group(2))
                    vib = float(match_alt.group(3)) if match_alt.group(3) is not None else 0.0
                    disp = float(match_alt.group(4))
                    print(f"\n[CRITICAL BREACH DETECTED FROM ESP32-S3] Node {node_id} triggered hardware threshold! (Tilt: {tilt} deg, Vib: {vib} g, Disp: {disp} mm)")

    except KeyboardInterrupt:
        print("\n[INFO] Closing serial bridge gracefully...")
        ser.close()

def run_prehardware_emulator(target_url="http://127.0.0.1:8000/api/hardware_telemetry"):
    """
    Simulates the exact output stream of central_hub_esp32s3_5.ino
    so you can test the entire hardware-to-dashboard-to-email pipeline
    before the physical hardware arrives.
    """
    print("\n=======================================================")
    print("[PRE-HARDWARE EMULATOR] RUNNING VIRTUAL ESP32-S3 TDMA HUB")
    print("Firmware Logic : central_hub_esp32s3_v6.ino")
    print("Wi-Fi Channel  : 1 (ESP-NOW Locked)")
    print("Superframe     : 1000ms (TDMA Slot 1: Node 1 | Slot 2: Node 2)")
    print(f"Target Backend : {target_url}")
    print("=======================================================")
    print("[TIP] Press Ctrl+C at any time to exit.\n")

    # Embedded 1D Discrete Kalman Filter matching firmware exactly
    class KalmanFilter1D:
        def __init__(self, q=0.02, r=1.5, p=1.0):
            self.q = q
            self.r = r
            self.p = p
            self.x = 0.0
            self.initialized = False

        def update(self, measurement):
            if not self.initialized:
                self.x = measurement
                self.initialized = True
                return self.x
            self.p = self.p + self.q
            denom = self.p + self.r
            k = (self.p / denom) if abs(denom) > 1e-6 else 0.5
            self.x = self.x + k * (measurement - self.x)
            self.p = (1.0 - k) * self.p
            return self.x

    kf_tilt = KalmanFilter1D(0.02, 1.5)
    kf_vib = KalmanFilter1D(0.05, 2.0)
    kf_disp = KalmanFilter1D(0.01, 0.8)

    frame_id = 0
    import random

    while True:
        try:
            frame_id += 1
            # Baseline stationary readings with slight physical noise
            raw_tilt = random.gauss(0.15, 0.25)
            raw_vib = abs(random.gauss(0.05, 0.04))
            raw_disp = max(0.0, random.gauss(0.8, 0.3))

            filt_tilt = kf_tilt.update(raw_tilt)
            filt_vib = kf_vib.update(raw_vib)
            filt_disp = kf_disp.update(raw_disp)

            sim_line = f"[NODE 1 KALMAN FILTERED] Tilt: {filt_tilt:+06.2f} deg | Vib: {filt_vib:05.2f} g | Disp: {filt_disp:05.1f} mm"
            print(f"[VIRTUAL SERIAL Frame #{frame_id:04d}] {sim_line}")

            status = "SAFE"
            payload = {
                "source": "VIRTUAL_ESP32_S3",
                "node_id": "NODE_01",
                "filter_mode": "KALMAN FILTERED",
                "tilt": round(filt_tilt, 2),
                "vibration": round(filt_vib, 2),
                "displacement": round(filt_disp, 1),
                "status": status,
                "frame_id": frame_id,
                "timestamp": time.strftime("%H:%M:%S")
            }
            post_telemetry_to_dashboard(payload, target_url)

            time.sleep(1.0) # 1000ms TDMA superframe
        except KeyboardInterrupt:
            print("\n[INFO] Virtual emulator stopped.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine Subsidence ESP32-S3 Hardware Bridge")
    parser.add_argument("--port", type=str, default=None, help="COM port for ESP32-S3 (e.g. COM3, COM4, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/api/hardware_telemetry", help="Target API URL")
    parser.add_argument("--mock", action="store_true", help="Run in Pre-Hardware Virtual TDMA Mode")
    args = parser.parse_args()

    if args.mock:
        run_prehardware_emulator(args.url)
    else:
        ports = list_available_ports()
        if args.port:
            run_hardware_listener(args.port, args.baud, args.url)
        elif len(ports) == 1:
            print(f"[AUTO-DETECT] Found device on port: {ports[0]}")
            run_hardware_listener(ports[0], args.baud, args.url)
        elif len(ports) > 1:
            print(f"Multiple COM ports detected: {', '.join(ports)}")
            selected = input(f"Enter target COM port [{ports[0]}]: ").strip()
            run_hardware_listener(selected if selected else ports[0], args.baud, args.url)
        else:
            print("[NOTICE] No physical USB COM ports currently detected.")
            print("         Starting Pre-Hardware Virtual TDMA Mode (matching central_hub_esp32s3_v6.ino math)...")
            run_prehardware_emulator(args.url)
