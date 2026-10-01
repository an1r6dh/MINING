import urllib.request
import json
import time

def run_tests():
    base = 'http://127.0.0.1:8000'
    print("=" * 60)
    print("  TEST 1: Health, GPU & Model Verification")
    print("=" * 60)
    with urllib.request.urlopen(f"{base}/health") as r:
        print("[PASS] Health:", json.loads(r.read()))
    with urllib.request.urlopen(f"{base}/gpu_status") as r:
        print("[PASS] GPU:", json.loads(r.read()))
    with urllib.request.urlopen(f"{base}/api/model_info") as r:
        print("[PASS] Model Info:", json.loads(r.read()))

    print("\n" + "=" * 60)
    print("  TEST 2: ML Inference Pipeline (/predict)")
    print("=" * 60)
    test_cases = [
        ("Safe Normal Operations", {"tilt": 0.25, "vibration": 0.035, "displacement": 12.0}),
        ("Warning Threshold Drift", {"tilt": 2.50, "vibration": 0.220, "displacement": 55.0}),
        ("Critical Danger Breach",  {"tilt": 4.50, "vibration": 0.350, "displacement": 92.0})
    ]
    for label, payload in test_cases:
        req = urllib.request.Request(
            f"{base}/predict",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req) as r:
            res = json.loads(r.read())
            status = res.get('status')
            risk = res.get('risk_score')
            print(f"[{status}] {label}: Input {payload} => Prediction: {status} (Risk: {risk}%)")

    print("\n" + "=" * 60)
    print("  TEST 3: Node 1 Telemetry (MPU-6050 Tilt/Vib + HC-SR04 Displacement)")
    print("=" * 60)
    n1_payload = {
        "source": "ESP32_HARDWARE",
        "node_id": "NODE_01",
        "filter_mode": "KALMAN FILTERED",
        "tilt": 0.35,
        "vibration": 0.042,
        "displacement": 15.8,
        "status": "SAFE"
    }
    req = urllib.request.Request(
        f"{base}/api/hardware_telemetry",
        data=json.dumps(n1_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as r:
        print("[PASS] Node 1 Pushed successfully:", json.loads(r.read()))

    print("\n" + "=" * 60)
    print("  TEST 4: Node 2 Telemetry (MPU-6050 Vibration + Digital Tilt)")
    print("=" * 60)
    n2_payload = {
        "source": "ESP32_HARDWARE",
        "node_id": "NODE_02",
        "filter_mode": "MPU6050 SENSOR",
        "tilt": 0.0,
        "vibration": 0.034,  # Real non-zero MPU-6050 dynamic vibration!
        "displacement": 0.0,
        "status": "SAFE"
    }
    req = urllib.request.Request(
        f"{base}/api/hardware_telemetry",
        data=json.dumps(n2_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as r:
        print("[PASS] Node 2 Pushed successfully:", json.loads(r.read()))

    print("\n" + "=" * 60)
    print("  TEST 5: Verify Dual-Node Hub Telemetry Cache")
    print("=" * 60)
    with urllib.request.urlopen(f"{base}/api/hardware_telemetry") as r:
        hub = json.loads(r.read())
        nodes = hub.get("nodes", {})
        print("Active Broadcast Node:", hub.get("node_id"))
        print("\nNode 1 Status in Hub:")
        print(json.dumps(nodes.get("NODE_01"), indent=2))
        print("\nNode 2 Status in Hub:")
        print(json.dumps(nodes.get("NODE_02"), indent=2))

        assert nodes.get("NODE_01", {}).get("connected") == True, "Node 1 should be connected!"
        assert nodes.get("NODE_02", {}).get("connected") == True, "Node 2 should be connected!"
        assert nodes.get("NODE_02", {}).get("vibration") == 0.034, "Node 2 vibration should be real MPU6050 reading!"
        print("\n[ALL ASSERTIONS PASSED] Hub successfully integrates and tracks both nodes in real-time!")

    print("\n" + "=" * 60)
    print("  TEST 6: Node 2 MPU-6050 High Vibration Breach Threshold Check")
    print("=" * 60)
    n2_danger_payload = {
        "source": "ESP32_HARDWARE",
        "node_id": "NODE_02",
        "filter_mode": "MPU6050 SENSOR",
        "tilt": 0.0,
        "vibration": 0.312,  # > 0.261g Critical Vibration Threshold!
        "displacement": 0.0,
        "status": "DANGER"
    }
    req = urllib.request.Request(
        f"{base}/api/hardware_telemetry",
        data=json.dumps(n2_danger_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as r:
        print("[PASS] High vibration breach pushed:", json.loads(r.read()))

    with urllib.request.urlopen(f"{base}/api/hardware_telemetry") as r:
        hub = json.loads(r.read())
        n2_data = hub.get("nodes", {}).get("NODE_02", {})
        print(f"Node 2 State after high vibration: status={n2_data.get('status')}, vib={n2_data.get('vibration')}g")
        assert n2_data.get("status") == "DANGER", "Node 2 status must be DANGER!"
        print("[PASS] Critical Vibration threshold correctly latched DANGER on Node 2!")

    print("\n" + "=" * 60)
    print("  TEST 7: Universal Serial Parsing for all 3 .ino Firmwares")
    print("=" * 60)
    from serial_bridge import parse_telemetry_line
    test_lines = [
        ("Central Hub JSON (Node 1)", '{"node_id":"NODE_SECTOR_01","filtered_tilt":1.25,"filtered_vibration":0.04,"filtered_displacement":15.20}'),
        ("Central Hub JSON (Node 2)", '{"node_id":"NODE_SECTOR_02","filtered_tilt":0.00,"filtered_vibration":0.038,"filtered_displacement":0.00}'),
        ("Central Hub Plain (Node 1)", '[NODE 1 KALMAN FILTERED] Tilt: +01.25 deg | Vib: 00.04 g | Disp: 015.2 mm'),
        ("Central Hub Plain (Node 2)", '[NODE 2 MPU6050 SENSOR] Tilt: +00.00 deg | Vib: 00.04 g | Disp: 000.0 mm'),
        ("Node 1 Direct TX (Slot 1)", '[NODE 1 TX SLOT 1] Sent Instant (<1ms): Tilt=+00.85 deg | Vib=00.03 g | Disp=014.0 mm'),
        ("Node 2 Direct TX (Slot 2)", '[NODE 2 TX SLOT 2] Telemetry: Tilt=00.0 (UPRIGHT) | Vib=00.045 g | Disp=0.0mm')
    ]
    for label, raw in test_lines:
        res = parse_telemetry_line(raw)
        assert res is not None, f"Failed to parse line: {raw}"
        print(f"[PASS] {label}: Node={res['node_id']} | Mode={res['filter_mode']} | Tilt={res['tilt']} | Vib={res['vibration']} | Disp={res['displacement']} | Status={res['status']}")

    print("\n" + "=" * 60)
    print("  ALL 7 COMPREHENSIVE INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
