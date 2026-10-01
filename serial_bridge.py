"""
====================================================================================================
PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
MODULE : USB Serial Hardware Bridge to Web Dashboard & Email Alert System
TARGET : Listens to ESP32-S3 Central Master Hub (central_hub_esp32s3_v7.ino) @ 115200 Baud
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
    Universal parser for all ESP32 firmware output formats:
    1. Central Hub JSON: {"node_id":"NODE_SECTOR_01","filtered_tilt":2.45,"filtered_vibration":0.12,"filtered_displacement":35.20}
    2. Central Hub Plain: [NODE 1 KALMAN FILTERED] Tilt: +00.00 deg | Vib: 00.00 g | Disp: 000.0 mm
    3. Central Hub Plain: [NODE 2 DIGITAL OVERRIDE] Tilt: +00.00 deg | Vib: 00.00 g | Disp: 000.0 mm
    4. Node 1 Direct: [NODE 1 TX SLOT 1] Sent Instant (<1ms): Tilt=+00.00 deg | Vib=00.00 g | Disp=000.0 mm
    5. Node 2 Direct: [NODE 2 TX SLOT 2] IMU Telemetry: Tilt=00.0 (UPRIGHT) | VibPeak=0.0000g (CALM) | Disp=0.0mm
    """
    # 1. JSON Telemetry line from Central Hub (streamJsonToLaptopML)
    if raw_line.startswith("{") and raw_line.endswith("}"):
        try:
            j = json.loads(raw_line)
            nid_raw = str(j.get("node_id", "1"))
            n_id = 2 if "2" in nid_raw else 1
            tilt = float(j.get("filtered_tilt", 0.0))
            vib = float(j.get("filtered_vibration", 0.0))
            disp = float(j.get("filtered_displacement", 0.0))
            filter_mode = "KALMAN FILTERED" if n_id == 1 else "DIGITAL OVERRIDE"

            abs_tilt = abs(tilt)
            if n_id == 2:
                is_danger = (abs_tilt >= 1.0 or vib >= 0.261 or disp >= 81.0)
                is_warning = False
            else:
                is_danger = (disp >= 81.0 or abs_tilt >= 3.801 or vib >= 0.261)
                is_warning = (disp >= 41.0 or abs_tilt >= 2.01 or vib >= 0.201)
            status = "DANGER" if is_danger else ("WARNING" if is_warning else "SAFE")

            return {
                "source": "ESP32_HARDWARE_BRIDGE",
                "node_id": f"NODE_0{n_id}",
                "filter_mode": filter_mode,
                "tilt": tilt,
                "vibration": vib,
                "displacement": disp,
                "status": status,
                "timestamp": time.strftime("%H:%M:%S")
            }
        except Exception:
            pass

    # 2. Universal regex parser
    node_m = re.search(r"NODE\s*(\d+)", raw_line, re.I)
    tilt_m = re.search(r"Tilt\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", raw_line, re.I)
    vib_m = re.search(r"Vib(?:Peak)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", raw_line, re.I)
    disp_m = re.search(r"Disp\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", raw_line, re.I)

    if tilt_m and (vib_m or disp_m):
        raw_upper = raw_line.upper()
        if node_m:
            node_id = int(node_m.group(1))
        elif "SLOT 2" in raw_upper or "NODE 2" in raw_upper:
            node_id = 2
        else:
            node_id = 1
        tilt = float(tilt_m.group(1))
        vib = float(vib_m.group(1)) if vib_m else 0.0
        disp = float(disp_m.group(1)) if disp_m else 0.0

        abs_tilt = abs(tilt)
        if node_id == 2:
            is_danger = (abs_tilt >= 1.0 or vib >= 0.261 or disp >= 81.0)
            is_warning = False
        else:
            is_danger = (disp >= 81.0 or abs_tilt >= 3.801 or vib >= 0.261)
            is_warning = (disp >= 41.0 or abs_tilt >= 2.01 or vib >= 0.201)
        status = "DANGER" if is_danger else ("WARNING" if is_warning else "SAFE")

        filter_type = "HARDWARE SENSOR"
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

# Regex matching Alert format from central_hub_esp32s3_v6.ino & v5.ino:
# "[LOCAL ALERT] Node 1 BREACH LATCHED! Tilt: 5.2 deg, Vib: 0.35 g, Disp: 42.0 mm"
# "[LOCAL ALERT] Node 1 breached safety limit! Tilt: 5.2 deg, Disp: 16.2 mm"
PATTERN_ALERT = re.compile(
    r"\[LOCAL ALERT\]\s+Node\s+(\d+)\s+(?:BREACH LATCHED!|breached safety limit!)\s+Tilt:\s*([+-]?\d+(?:\.\d+)?)\s*deg(?:,\s*Vib:\s*([+-]?\d+(?:\.\d+)?)\s*g)?,\s*Disp:\s*([+-]?\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE
)

def post_telemetry_to_dashboard(payload, target_url="http://127.0.0.1:8000/api/hardware_telemetry", token=None):
    """Posts live parsed hardware readings to the dashboard backend."""
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        if token:
            headers["x-vercel-protection-bypass"] = token
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            target_url,
            data=data_bytes,
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False

def list_available_ports():
    """Returns a list of all detected COM ports, excluding Bluetooth phantom ports."""
    if not HAS_SERIAL:
        return []
    valid_ports = []
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        if "bthenum" in hwid or "bluetooth" in desc or "bluetooth" in hwid:
            continue
        valid_ports.append(port.device)
    return valid_ports

def run_hardware_listener(port, baudrate=115200, target_url="http://127.0.0.1:8000/api/hardware_telemetry", token=None):
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
                        posted = post_telemetry_to_dashboard(payload, target_url, token=token)
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

def run_prehardware_emulator(target_url="http://127.0.0.1:8000/api/hardware_telemetry", once=False, token=None):
    """
    Simulates the exact output stream of central_hub_esp32s3_v7.ino
    so you can test the entire hardware-to-dashboard-to-email pipeline
    before the physical hardware arrives.
    """
    print("\n=======================================================")
    print("[PRE-HARDWARE EMULATOR] RUNNING VIRTUAL ESP32-S3 TDMA HUB")
    print("Firmware Logic : central_hub_esp32s3_v7.ino")
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

            # ---------------------------------------------------------
            # 1. SLOT 1: NODE 1 (Geotechnical Continuous - Kalman Filtered)
            # ---------------------------------------------------------
            raw_tilt1 = random.gauss(0.15, 0.25)
            raw_vib1 = abs(random.gauss(0.05, 0.04))
            raw_disp1 = max(0.0, random.gauss(0.8, 0.3))

            filt_tilt1 = kf_tilt.update(raw_tilt1)
            filt_vib1 = kf_vib.update(raw_vib1)
            filt_disp1 = kf_disp.update(raw_disp1)

            sim_line1 = f"[NODE 1 KALMAN FILTERED] Tilt: {filt_tilt1:+06.2f} deg | Vib: {filt_vib1:05.2f} g | Disp: {filt_disp1:05.1f} mm"
            json_line1 = json.dumps({
                "node_id": "NODE_SECTOR_01",
                "filtered_tilt": round(filt_tilt1, 2),
                "filtered_vibration": round(filt_vib1, 2),
                "filtered_displacement": round(filt_disp1, 1)
            })
            print(f"[VIRTUAL SERIAL Slot 1] {sim_line1}")
            print(f"                      {json_line1}")

            payload1 = {
                "source": "VIRTUAL_ESP32_S3",
                "node_id": "NODE_01",
                "filter_mode": "KALMAN FILTERED",
                "tilt": round(filt_tilt1, 2),
                "vibration": round(filt_vib1, 2),
                "displacement": round(filt_disp1, 1),
                "status": "SAFE",
                "frame_id": frame_id,
                "timestamp": time.strftime("%H:%M:%S")
            }
            posted1 = post_telemetry_to_dashboard(payload1, target_url, token=token)
            print(f"       --> PUSHED NODE_01: {payload1['tilt']:+.2f}deg | {payload1['vibration']:.2f}g | {payload1['displacement']:.1f}mm | {'[OK]' if posted1 else '[OFFLINE]'}")

            time.sleep(0.08) # 80ms slot 1 duration

            # ---------------------------------------------------------
            # 2. SLOT 2: NODE 2 (Perimeter Digital Tripwire - SW-420 + Tilt)
            # ---------------------------------------------------------
            n2_tilt = 0.0
            n2_vib = 0.0
            n2_disp = 0.0

            sim_line2 = f"[NODE 2 DIGITAL OVERRIDE] Tilt: {n2_tilt:+06.2f} deg | Vib: {n2_vib:05.2f} g | Disp: {n2_disp:05.1f} mm"
            json_line2 = json.dumps({
                "node_id": "NODE_SECTOR_02",
                "filtered_tilt": n2_tilt,
                "filtered_vibration": n2_vib,
                "filtered_displacement": n2_disp
            })
            print(f"[VIRTUAL SERIAL Slot 2] {sim_line2}")
            print(f"                      {json_line2}")

            payload2 = {
                "source": "VIRTUAL_ESP32_S3",
                "node_id": "NODE_02",
                "filter_mode": "DIGITAL OVERRIDE",
                "tilt": n2_tilt,
                "vibration": n2_vib,
                "displacement": n2_disp,
                "status": "SAFE",
                "frame_id": frame_id,
                "timestamp": time.strftime("%H:%M:%S")
            }
            posted2 = post_telemetry_to_dashboard(payload2, target_url, token=token)
            print(f"       --> PUSHED NODE_02: {payload2['tilt']:+.2f}deg | {payload2['vibration']:.2f}g | {payload2['displacement']:.1f}mm | {'[OK]' if posted2 else '[OFFLINE]'}")

            if once:
                break

            time.sleep(0.92) # Complete the 1000ms TDMA superframe
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
    for priority_port in ["COM8", "COM7"]:
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
    parser.add_argument("--url", "--target", dest="url", type=str, default="http://127.0.0.1:8000/api/hardware_telemetry", help="Target API URL")
    parser.add_argument("--token", type=str, default=os.getenv("VERCEL_PROTECTION_TOKEN", None), help="Vercel Protection Bypass Token")
    parser.add_argument("--vercel", action="store_true", help="Stream telemetry directly to live Vercel dashboard")
    parser.add_argument("--mock", action="store_true", help="Run in Pre-Hardware Virtual TDMA Mode")
    parser.add_argument("--once", action="store_true", help="Emit a single TDMA superframe test packet and exit")
    args = parser.parse_args()

    target_url = "https://mining-anivedas-5643s-projects.vercel.app/api/hardware_telemetry" if args.vercel else args.url

    if args.mock or args.once:
        run_prehardware_emulator(target_url, once=args.once, token=args.token)
    else:
        ports = list_available_ports()
        if args.port:
            run_hardware_listener(args.port, args.baud, target_url, token=args.token)
        elif ports:
            target_port = auto_detect_esp32_port(ports, args.baud)
            print(f"[BRIDGE] Selected active port: {target_port}")
            run_hardware_listener(target_port, args.baud, target_url, token=args.token)
        else:
            print("[NOTICE] No physical USB COM ports currently detected.")
            print("         Starting Pre-Hardware Virtual TDMA Mode (matching central_hub_esp32s3_v7.ino math)...")
            run_prehardware_emulator(target_url, once=args.once, token=args.token)
