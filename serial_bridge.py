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

def parse_telemetry_line(raw_line):
    """
    Universal parser for all 3 ESP32 firmware output formats:
    1. Central Hub: [NODE 1 KALMAN FILTERED] Tilt: +00.00 deg | Vib: 00.00 g | Disp: 000.0 mm
    2. Central Hub: [NODE 2 DIGITAL OVERRIDE] Tilt: +00.00 deg | Vib: 00.00 g | Disp: 000.0 mm
    3. Node 1 Direct: [NODE 1 TX SLOT 1] Sent Instant (<1ms): Tilt=+00.00 deg | Vib=00.00 g | Disp=000.0 mm
    4. Node 2 Direct: [NODE 2 TX SLOT 2] Digital Telemetry: Tilt=00.0 deg (NORMAL) | Vib=00.0 (NORMAL) | Disp=0.0mm
    """
    node_m = re.search(r"NODE\s*(\d+)", raw_line, re.I)
    tilt_m = re.search(r"Tilt\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", raw_line, re.I)
    vib_m = re.search(r"Vib\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", raw_line, re.I)
    disp_m = re.search(r"Disp\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", raw_line, re.I)

    if tilt_m and (vib_m or disp_m):
        node_id = int(node_m.group(1)) if node_m else 1
        tilt = float(tilt_m.group(1))
        vib = float(vib_m.group(1)) if vib_m else 0.0
        disp = float(disp_m.group(1)) if disp_m else 0.0

        is_danger = (tilt >= 5.0 or vib >= 0.8 or disp >= 15.0)
        is_warning = (tilt >= 2.5 or vib >= 0.4 or disp >= 8.0)
        status = "DANGER" if is_danger else ("WARNING" if is_warning else "SAFE")

        filter_type = "HARDWARE SENSOR"
        raw_upper = raw_line.upper()
        if "KALMAN" in raw_upper:
            filter_type = "KALMAN FILTERED"
        elif "OVERRIDE" in raw_upper or "DIGITAL" in raw_upper:
            filter_type = "DIGITAL OVERRIDE"
        elif "SLOT 1" in raw_upper:
            filter_type = "NODE 1 DIRECT (Slot 1)"
        elif "SLOT 2" in raw_upper:
            filter_type = "NODE 2 DIRECT (Slot 2)"

        return {
            "source": "ESP32_HARDWARE_BRIDGE",
            "node_id": f"NODE_0{node_id}",
            "filter_mode": filter_type,
            "tilt": tilt,
            "vibration": vib,
            "displacement": disp,
            "status": status,
            "timestamp": time.strftime("%H:%M:%S")
        }
    return None

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
    print(f"[BRIDGE] OPENING USB SERIAL CONNECTION TO HARDWARE")
    print(f"Port      : {port}")
    print(f"Baud Rate : {baudrate}")
    print(f"Backend   : {target_url}")
    print(f"=======================================================")

    while True:
        ser = None
        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=1.0)
            print(f"[SUCCESS] Connected to {port}! Listening for live ESP-NOW TDMA frames...\n")
            while True:
                if ser.in_waiting > 0:
                    raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw_line:
                        continue

                    print(f"[RX] {raw_line}")

                    # 1. Check for standard telemetry line (universal parser)
                    payload = parse_telemetry_line(raw_line)
                    if payload:
                        posted = post_telemetry_to_dashboard(payload, target_url)
                        indicator = "[WEB SYNC OK]" if posted else "[LOCAL ONLY]"
                        print(f"       --> PUSHED {payload['node_id']}: Tilt={payload['tilt']:+.2f}deg | Vib={payload['vibration']:.2f}g | Disp={payload['displacement']:.1f}mm | {payload['status']} {indicator}")

                    # 2. Check for local alert line
                    match_alt = PATTERN_ALERT.search(raw_line)
                    if match_alt:
                        node_id = int(match_alt.group(1))
                        tilt = float(match_alt.group(2))
                        vib = float(match_alt.group(3)) if match_alt.group(3) is not None else 0.0
                        disp = float(match_alt.group(4))
                        print(f"\n[CRITICAL BREACH DETECTED FROM HARDWARE] Node {node_id} triggered threshold! (Tilt: {tilt} deg, Vib: {vib} g, Disp: {disp} mm)")
                else:
                    time.sleep(0.04)

        except KeyboardInterrupt:
            print("\n[INFO] Closing serial bridge gracefully...")
            if ser and ser.is_open:
                ser.close()
            break
        except Exception as e:
            print(f"[RECONNECT] Serial connection error ({e}). Retrying in 2 seconds...")
            if ser:
                try: ser.close()
                except Exception: pass
            time.sleep(2.0)

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

def auto_detect_esp32_port(ports, baudrate=115200):
    """Probes detected COM ports to find which one is streaming ESP32 hardware telemetry."""
    if not ports:
        return None
    if len(ports) == 1:
        return ports[0]
    
    # Priority for known active ports
    for priority_port in ["COM7", "COM8"]:
        if priority_port in ports:
            print(f"[AUTO-DETECT] Prioritizing active hardware port on {priority_port}")
            return priority_port

    print(f"[PROBING] Multiple COM ports detected: {', '.join(ports)}. Probing for ESP32 stream...")
    for p in ports:
        try:
            print(f"  --> Checking {p}...")
            ser = serial.Serial(p, baudrate=baudrate, timeout=1.2)
            start = time.time()
            found = False
            while time.time() - start < 1.5:
                if ser.in_waiting:
                    line = ser.readline().decode("utf-8", errors="ignore")
                    if parse_telemetry_line(line) or "[NODE" in line or "[SYSTEM]" in line or "ESP32" in line or "[READY]" in line:
                        found = True
                        break
            ser.close()
            if found:
                print(f"[FOUND] Active ESP32 device identified on {p}!")
                return p
        except Exception:
            pass
    return ports[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine Subsidence ESP32-S3 Hardware Bridge")
    parser.add_argument("--port", type=str, default=None, help="COM port for ESP32-S3 (e.g. COM7, COM3, /dev/ttyUSB0)")
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
        elif ports:
            target_port = auto_detect_esp32_port(ports, args.baud)
            print(f"[BRIDGE] Selected active port: {target_port}")
            run_hardware_listener(target_port, args.baud, args.url)
        else:
            print("[NOTICE] No physical USB COM ports currently detected.")
            print("         Starting Pre-Hardware Virtual TDMA Mode (matching central_hub_esp32s3_5.ino math)...")
            run_prehardware_emulator(args.url)
