import urllib.request

with urllib.request.urlopen('http://127.0.0.1:8000/') as r:
    html = r.read().decode('utf-8')

checks = [
    'Slot 2: <strong style="color: #f59e0b;">Node 2 (MPU6050 IMU)</strong>',
    'MPU-6050 Accel (Node 2 Live)',
    '0.000 mm • Ultrasonic on Node 1 (Offline)',
    'Sensor Node 2 — MPU-6050 Surface / Slope IMU',
    '6-DOF IMU Telemetry (TDMA Slot 2)',
    'parseAndApplyHardwareLine',
    'updateTelemetry',
    'syncNodeConnectionStatus'
]

print("=== DASHBOARD HTML CONTENT VERIFICATION ===")
for c in checks:
    if c in html:
        print(f"[FOUND] {c[:65]}")
    else:
        print(f"[MISSING] {c}")
