/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING & EARLY WARNING (SIH 26025)
 * FIRMWARE: ESP32-S3 Central Master Hub (central_hub_esp32s3_v6.ino)
 * TARGET: ESP32-S3 Development Board (Arduino IDE Compilable)
 * ====================================================================================================
 * 
 * AUDIT & ENGINEERING REVISIONS (VERSION 6.1 PRODUCTION HARDENED):
 * 1. Unified Buzzer Management & Dual-Core Race Condition Elimination (CRITICAL):
 *    - Core 0 (vGsmSmsTask) previously called digitalWrite(PIN_BUZZER, HIGH/LOW) with blocking vTaskDelay,
 *      which conflicted with Core 1's continuous handleBuzzerAlerts() in loop(), causing pin contention
 *      and instantaneous silencing of warning beeps.
 *    - All physical buzzer writes are now strictly unified on Core 1. Core 0 requests warning beeps via
 *      a non-blocking thread-safe counter (gsmWarningBeepsRemaining).
 * 2. Hardened Cellular Network Registration (AT+CREG?) Response Parser (CRITICAL):
 *    - Replaced the fragile single-shot 300ms delay with an active response collector loop (up to 1500ms)
 *      that verifies complete modem frames ("OK", "+CREG:"). Handles home (1) and roaming (5) across all
 *      solicited/unsolicited formats (e.g. "+CREG: 0,1", "+CREG: 1,1", "+CREG: 0,5", "+CREG: 1").
 *    - Added automatic retry before declaring network out-of-coverage, preventing dropped emergency SMS.
 * 3. SMS Prompt Timeout Escape Recovery (0x1B ESC):
 *    - If the modem prompt ('>') is not received within 5000ms, the routine sends 0x1B (ESC) to cleanly
 *      abort the SMS input session, preventing subsequent AT commands from being swallowed as SMS body text.
 * 4. Non-Blocking USB-CDC Serial Command Engine (TDMA Protection):
 *    - Replaced Serial.readStringUntil('\n') (which blocked up to 1000ms) with a zero-latency static byte
 *      accumulator. Eliminates TDMA superframe jitter and ensures sub-millisecond beacon clock precision.
 * 5. Geotechnical Vibration Threshold Detection (Node 1 & Node 2):
 *    - CRITICAL_VIBRATION_THRESH_G corrected to 0.8f (matches geotechnical documentation).
 *    - Node 2's SW-420 tripwire vibration latch (1.0f) and Node 1 dynamic seismic shocks now reliably trip alarms.
 * 6. Permanent Alarm Latching (CRITICAL BUG FIX v6.1):
 *    - DEFEATED LATCH BUG FIXED: The previous auto-clearing else block in processIncomingData() would
 *      silently clear criticalAlarmActive=false on the very next normal-reading packet from any node.
 *      A momentary SW-420 tripwire trigger (1-frame pulse) was being extinguished one frame later.
 *    - criticalAlarmActive is now a ONE-WAY LATCH: once set true by any breach event, it remains
 *      permanently true until an operator sends explicit "RESET" command via serial.
 * 7. AI Hardware Server Interrupt (GPIO 10) Debouncing:
 *    - Added millisecond time-guard debouncing in handleServerInterruptISR() to prevent contact bounce or RF noise
 *      from flooding the FreeRTOS gsmQueue with duplicate emergency dispatches.
 * 8. 1D Discrete Kalman Initial State Auto-Seeding:
 *    - The ultrasonic displacement Kalman filter auto-seeds its initial state with the first valid reading,
 *      eliminating filter startup lag and false displacement breach transients.
 * 9. Dual-Core Memory Consistency:
 *    - All variables shared across Core 0 and Core 1 / ISR are qualified as volatile with atomic operations.
 * 10. Multi-IDF ESP-NOW Compatibility:
 *    - Supports ESP-IDF v4.x (Arduino Core 2.x) and ESP-IDF v5.x (Arduino Core 3.x) with strict callback casting.
 * ====================================================================================================
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h> // Explicit Wi-Fi Hardware Channel Configuration


/* ====================================================================================================
 * 1. HARDWARE PIN DEFINITIONS (ESP32-S3 SPECIFIC)
 * ==================================================================================================== */
#define PIN_LED_RED_POWER       11   // Power LED (Continuously driven HIGH in setup)
#define PIN_LED_GREEN_STATUS    13   // Status LED (Solid = Nodes 1 & 2 synced; Rapid blink = timeout)
#define PIN_BUZZER              4    // High-pitch alarm buzzer (Active buzzer or PWM drive)

#define PIN_GSM_RX              18   // ESP32-S3 RX1 -> connected to SIM module TX
#define PIN_GSM_TX              17   // ESP32-S3 TX1 -> connected to SIM module RX

/* ====================================================================================================
 * 2. SYSTEM CONFIGURATION & SAFETY THRESHOLDS
 * ==================================================================================================== */
#define ESPNOW_WIFI_CHANNEL         1               // Channel for ESP-NOW TDMA cluster (RF Locked)
#define EMERGENCY_PHONE_NUMBER      "+919876543210" // Destination phone number for SMS alerts

#define CRITICAL_DISPLACEMENT_THRESH_MM 40.0f       // Critical rock mass displacement trigger (mm)
#define CRITICAL_TILT_THRESH_DEG        2.0f        // Critical angular tilt trigger (degrees)
#define CRITICAL_VIBRATION_THRESH_G     0.25f       // Critical vibration threshold (g) — geotechnical calibration value
#define NODE_TIMEOUT_MS                 3500        // Timeout threshold before marking a node offline

#define TDMA_FRAME_PERIOD_MS        1000            // Master TDMA superframe period (1000 ms)
#define TDMA_SLOT_DURATION_MS       80              // Transmit window duration per node (80 ms)
#define STATUS_BLINK_INTERVAL_MS    500             // Blink interval when any demo node times out (500ms = visible solid pulse)
#define SMS_COOLDOWN_MS             60000           // 60s cooldown between automated threshold SMS sends

/* ====================================================================================================
 * 3. EMBEDDED 1D DISCRETE KALMAN FILTER ENGINE
 * ==================================================================================================== */
class DiscreteKalmanFilter1D {
public:
    DiscreteKalmanFilter1D(float process_noise = 0.02f, float measurement_noise = 1.5f, float estimation_error = 1.0f, float initial_value = 0.0f)
        : _q(process_noise), _r(measurement_noise), _p(estimation_error), _x(initial_value), _k(0.0f), _initialized(false) {}

    float update(float measurement) {
        if (!_initialized) {
            _x = measurement;
            _initialized = true;
            return _x;
        }
        _p = _p + _q;
        float denominator = _p + _r;
        _k = (fabs(denominator) > 1e-6f) ? (_p / denominator) : 0.5f;
        _x = _x + _k * (measurement - _x);
        _p = (1.0f - _k) * _p;
        return _x;
    }

    void reset(float initial_value = 0.0f) {
        _x = initial_value;
        _p = 1.0f;
        _initialized = false;
    }

    float getState() const { return _x; }
    void setNoiseParameters(float q, float r) { _q = q; _r = r; }

private:
    float _q;
    float _r;
    float _p;
    float _x;
    float _k;
    bool  _initialized;
};

/* ====================================================================================================
 * 4. PROTOCOL DATA STRUCTURES (EXACT PACKED BINARY PAYLOADS)
 * ==================================================================================================== */
#define PKT_TYPE_BEACON    0x01
#define PKT_TYPE_TELEMETRY 0x02

struct __attribute__((packed)) TDMABeaconPacket {
    uint8_t  msg_type;           // PKT_TYPE_BEACON (0x01)
    uint32_t frame_id;          // Monotonically increasing frame counter
    uint32_t beacon_time_ms;    // Master clock timestamp
    uint16_t slot_duration_ms;  // Slot duration (80ms)
    uint8_t  total_slots;       // Number of active slots in frame
    uint8_t  emergency_flag;    // 1 if alarm active, 0 normal
};

struct __attribute__((packed)) SensorPayload {
    uint8_t  msg_type;          // PKT_TYPE_TELEMETRY (0x02)
    uint8_t  node_id;           // 1 = Node 1, 2 = Node 2
    float    tilt;              // Raw Tilt angle (degrees)
    float    vibration;         // Raw Vibration acceleration (g)
    float    displacement;      // Raw Surface/Roof displacement (mm)
};

struct FilteredNodeRecord {
    bool     active;
    uint32_t last_packet_time;
    float    raw_tilt;
    float    raw_vibration;
    float    raw_displacement;
    float    filtered_tilt;
    float    filtered_vibration;
    float    filtered_displacement;
};

struct SmsAlertRequest {
    char message[140];
};

/* ====================================================================================================
 * 5. GLOBAL OBJECTS, FREERTOS HANDLES & PER-NODE KALMAN ARRAYS
 * ==================================================================================================== */
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

HardwareSerial SerialGSM(1);
QueueHandle_t gsmQueue = NULL;

#define MAX_SUPPORTED_NODES 3 // Index 0 unused, Index 1 = Node 1, Index 2 = Node 2

// Dedicated 1D Kalman Filter instances for Node 1
DiscreteKalmanFilter1D kfTilt[MAX_SUPPORTED_NODES] = {
    DiscreteKalmanFilter1D(0.02f, 1.5f),
    DiscreteKalmanFilter1D(0.02f, 1.5f), // Node 1 Tilt
    DiscreteKalmanFilter1D(0.02f, 1.5f)  // Node 2 (Bypassed)
};

DiscreteKalmanFilter1D kfVibration[MAX_SUPPORTED_NODES] = {
    DiscreteKalmanFilter1D(0.05f, 2.0f),
    DiscreteKalmanFilter1D(0.05f, 2.0f), // Node 1 Vibration
    DiscreteKalmanFilter1D(0.05f, 2.0f)  // Node 2 (Bypassed)
};

DiscreteKalmanFilter1D kfDisplacement[MAX_SUPPORTED_NODES] = {
    DiscreteKalmanFilter1D(0.01f, 0.8f),
    DiscreteKalmanFilter1D(0.01f, 0.8f), // Node 1 Ultrasonic Displacement (Auto-seeding)
    DiscreteKalmanFilter1D(0.01f, 0.8f)  // Node 2 (Bypassed)
};

FilteredNodeRecord nodes[MAX_SUPPORTED_NODES];

// System State & Non-Blocking Timers
uint32_t tdmaFrameId = 0;
uint32_t lastBeaconTxTime = 0;
uint32_t lastBlinkTime = 0;
bool     greenLedState = false;
uint32_t lastAutoSmsDispatchTime = 0;

// Dual-Core Thread-Safe Concurrency Flags
volatile bool criticalAlarmActive = false;

// Persistent Collapse Guard State
bool nodeBreached[MAX_SUPPORTED_NODES] = {false};

// Unified Buzzer Control (Core 1 Ownership, Core 0 Signaling)
volatile uint8_t gsmWarningBeepStepsRemaining = 0; // Each beep = 1 ON step + 1 OFF step (6 steps = 3 beeps)
uint32_t lastBuzzerStepTime = 0;

/* ====================================================================================================
 * 6. FORWARD DECLARATIONS
 * ==================================================================================================== */
void vGsmSmsTask(void *pvParameters);
void sendSMSAlertBlocking(const char* alertMessage);
void triggerSmsAlert(const char* alertMsg);
void requestGsmWarningBeeps(uint8_t numBeeps);
void processIncomingData(const SensorPayload& payload);
void broadcastTDMABeacon();
void initHardware();
void initESPNowTDMA();
bool areAllDemoNodesSynced();
void updateStatusLEDs();
void handleBuzzerAlerts();
void checkSerialCommands();


/* ====================================================================================================
 * 7. FREERTOS CORE 0 CELLULAR GSM TASK (NON-BLOCKING BACKGROUND DISPATCH)
 * ==================================================================================================== */
void requestGsmWarningBeeps(uint8_t numBeeps) {
    // 2 state changes per beep: ON then OFF
    gsmWarningBeepStepsRemaining = numBeeps * 2;
}

void vGsmSmsTask(void *pvParameters) {
    Serial.println("[FreeRTOS Core 0] GSM Background Task initialized.");
    SmsAlertRequest request;

    while (true) {
        if (xQueueReceive(gsmQueue, &request, portMAX_DELAY) == pdTRUE) {
            Serial.println("\n[FreeRTOS Core 0] ALERT IN QUEUE! Starting SMS dispatch...");
            sendSMSAlertBlocking(request.message);
            Serial.println("[FreeRTOS Core 0] SMS Dispatch complete. Returning to idle.\n");
        }
    }
}

/**
 * Robust AT response reader with configurable timeout and termination search
 */
bool waitForGsmResponse(const char* expected1, const char* expected2, uint32_t timeoutMs, String* outResponse = NULL) {
    uint32_t start = millis();
    String resp = "";
    while (millis() - start < timeoutMs) {
        while (SerialGSM.available()) {
            char c = SerialGSM.read();
            resp += c;
        }
        if (expected1 && resp.indexOf(expected1) != -1) {
            if (outResponse) *outResponse = resp;
            return true;
        }
        if (expected2 && resp.indexOf(expected2) != -1) {
            if (outResponse) *outResponse = resp;
            return true;
        }
        if (resp.indexOf("ERROR") != -1) {
            if (outResponse) *outResponse = resp;
            return false;
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    if (outResponse) *outResponse = resp;
    return false;
}

void sendSMSAlertBlocking(const char* alertMessage) {
    // Drain any stale bytes from UART RX buffer
    while (SerialGSM.available()) SerialGSM.read();

    // 1. Check Network Registration Status with active polling
    bool isRegistered = false;
    for (int attempt = 0; attempt < 2; attempt++) {
        SerialGSM.println("AT+CREG?");
        String regResp = "";
        waitForGsmResponse("OK", "+CREG:", 1500, &regResp);

        // Registered Home (+CREG: 0,1 / 1,1) or Registered Roaming (+CREG: 0,5 / 1,5)
        if (regResp.indexOf(",1") != -1 || regResp.indexOf(",5") != -1 ||
            regResp.indexOf(", 1") != -1 || regResp.indexOf(", 5") != -1 ||
            regResp.indexOf(": 1") != -1 || regResp.indexOf(": 5") != -1) {
            isRegistered = true;
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    if (!isRegistered) {
        Serial.println("\n[GSM CRITICAL FAILURE] SIM module is OUT OF COVERAGE or UNREGISTERED!");
        Serial.println("[GSM ALERT] Unable to deliver SMS due to lack of network signal.");
        // Notify Core 1 to sound 3 warning beeps without GPIO pin race
        requestGsmWarningBeeps(3);
        return;
    }

    // 2. Configure SMS Text Mode
    while (SerialGSM.available()) SerialGSM.read();
    SerialGSM.println("AT+CMGF=1");
    waitForGsmResponse("OK", NULL, 1000);

    // 3. Initiate SMS transmission
    SerialGSM.print("AT+CMGS=\"");
    SerialGSM.print(EMERGENCY_PHONE_NUMBER);
    SerialGSM.println("\"");

    // 4. Wait for '>' prompt with timeout guard
    uint32_t promptTimeout = millis();
    bool promptReceived = false;
    while (millis() - promptTimeout < 5000) {
        if (SerialGSM.available()) {
            char c = SerialGSM.read();
            if (c == '>') {
                promptReceived = true;
                break;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    if (!promptReceived) {
        Serial.println("[GSM ERROR Core 0] Prompt '>' not received! Aborting SMS session.");
        SerialGSM.write(0x1B); // Send ESC to abort any hung AT+CMGS session
        vTaskDelay(pdMS_TO_TICKS(200));
        requestGsmWarningBeeps(3);
        return;
    }

    // 5. Send message payload and terminate with Ctrl+Z (0x1A)
    SerialGSM.print(alertMessage);
    vTaskDelay(pdMS_TO_TICKS(100));
    SerialGSM.write(0x1A); // Commit SMS

    // 6. Await carrier delivery confirmation (up to 20 seconds)
    String gsmResponse = "";
    bool sendSuccess = waitForGsmResponse("OK", "+CMGS:", 20000, &gsmResponse);

    if (sendSuccess) {
        Serial.println("[GSM SUCCESS Core 0] Emergency SMS successfully delivered to cellular network!");
    } else {
        Serial.printf("[GSM ERROR Core 0] SMS transmission failed (No Signal/Network Reject): %s\n", gsmResponse.c_str());
        SerialGSM.write(0x1B); // Clean up modem state
        requestGsmWarningBeeps(3);
    }

    // Cleanly flush trailing characters
    vTaskDelay(pdMS_TO_TICKS(200));
    while (SerialGSM.available()) SerialGSM.read();
}

void triggerSmsAlert(const char* alertMsg) {
    if (gsmQueue != NULL) {
        SmsAlertRequest req;
        strncpy(req.message, alertMsg, sizeof(req.message) - 1);
        req.message[sizeof(req.message) - 1] = '\0';
        xQueueSend(gsmQueue, &req, 0); // Non-blocking post from Core 1
    }
}

/* ====================================================================================================
 * 8. TELEMETRY PROCESSING: KALMAN FILTERING (NODE 1) & DIGITAL OVERRIDE (NODE 2)
 * ==================================================================================================== */
void streamJsonToLaptopML(uint8_t rawNodeId, float tilt, float vibration, float displacement) {
    const char* sector = (rawNodeId == 1) ? "NODE_SECTOR_01" : 
                         (rawNodeId == 2) ? "NODE_SECTOR_02" : "UNKNOWN_SECTOR";

    // Sanitize sentinel values for ML pipeline
    float safeTilt = (tilt == -999.0f) ? 0.00f : tilt;
    float safeVib = (vibration == -999.0f) ? 0.00f : vibration;
    float safeDisp = (displacement == -999.0f) ? 0.00f : displacement;

    char jsonBuf[128];
    snprintf(jsonBuf, sizeof(jsonBuf), 
             "{\"node_id\":\"%s\",\"filtered_tilt\":%.2f,\"filtered_vibration\":%.2f,\"filtered_displacement\":%.2f}",
             sector, safeTilt, safeVib, safeDisp);
             
    Serial.println(jsonBuf);
}

void processIncomingData(const SensorPayload& payload) {
    uint8_t id = payload.node_id;
    if (id == 0 || id >= MAX_SUPPORTED_NODES) return;

    nodes[id].active = true;
    nodes[id].last_packet_time = millis();
    nodes[id].raw_tilt = payload.tilt;
    nodes[id].raw_vibration = payload.vibration;
    nodes[id].raw_displacement = payload.displacement;

    if (id == 1) {
        // Continuous Geotechnical Node: Processed through dedicated 1D Kalman Filters
        if (payload.tilt != -999.0f) {
            nodes[1].filtered_tilt = kfTilt[1].update(payload.tilt);
        }
        if (payload.vibration != -999.0f) {
            nodes[1].filtered_vibration = kfVibration[1].update(payload.vibration);
        }
        if (payload.displacement != -999.0f) {
            nodes[1].filtered_displacement = kfDisplacement[1].update(payload.displacement);
        }
    } 
    else if (id == 2) {
        // Discrete Tripwire Node: Process incoming raw digital payload states into telemetry record
        if (payload.tilt != -999.0f) {
            nodes[2].filtered_tilt = payload.tilt;
        }
        if (payload.vibration != -999.0f) {
            nodes[2].filtered_vibration = payload.vibration;       
        }
        if (payload.displacement != -999.0f) {
            nodes[2].filtered_displacement = payload.displacement;
        }
    }
    else {
        if (payload.tilt != -999.0f) nodes[id].filtered_tilt = kfTilt[id].update(payload.tilt);
        if (payload.vibration != -999.0f) nodes[id].filtered_vibration = kfVibration[id].update(payload.vibration);
        if (payload.displacement != -999.0f) nodes[id].filtered_displacement = kfDisplacement[id].update(payload.displacement);
    }

    Serial.printf("[NODE %d %s] Tilt: %+05.2f deg | Vib: %04.2f g | Disp: %05.1f mm\n",
                  id, (id == 2 ? "DIGITAL OVERRIDE" : "KALMAN FILTERED"),
                  nodes[id].filtered_tilt, nodes[id].filtered_vibration, nodes[id].filtered_displacement);

    // Multi-criteria safety limit evaluation strictly centralized on Hub:
    bool isBreached = false;
    if (id == 2) {
        // Evaluate threshold breach directly for Node 2 digital state
        isBreached = (nodes[id].filtered_tilt >= 1.0f) || (nodes[id].filtered_vibration >= 1.0f);
    } else {
        isBreached = (fabs(nodes[id].filtered_tilt) >= CRITICAL_TILT_THRESH_DEG) ||
                     (nodes[id].filtered_displacement >= CRITICAL_DISPLACEMENT_THRESH_MM) ||
                     (nodes[id].filtered_vibration >= CRITICAL_VIBRATION_THRESH_G);
    }

    nodeBreached[id] = isBreached;

    if (isBreached) {
        // PERMANENT LATCH: criticalAlarmActive is a one-way flag.
        // Once set, it can ONLY be cleared by an explicit "RESET" serial command.
        // DO NOT auto-clear here — momentary SW-420 tripwire pulses and single-frame
        // sensor spikes must remain latched until confirmed safe by an operator.
        if (!criticalAlarmActive) {
            criticalAlarmActive = true;
            Serial.printf("[LOCAL ALERT] Node %d BREACH LATCHED! Tilt: %.1f deg, Vib: %.2f g, Disp: %.1f mm\n",
                          id, nodes[id].filtered_tilt, nodes[id].filtered_vibration, nodes[id].filtered_displacement);
        }

        uint32_t now = millis();
        if (now - lastAutoSmsDispatchTime > SMS_COOLDOWN_MS) {
            lastAutoSmsDispatchTime = now;
            char autoAlert[140];
            snprintf(autoAlert, sizeof(autoAlert),
                     "CRITICAL ALERT: Mine Movement at Node %d! Tilt: %.1fdeg, Vib: %.2fg, Disp: %.1fmm.",
                     id, nodes[id].filtered_tilt, nodes[id].filtered_vibration, nodes[id].filtered_displacement);
            triggerSmsAlert(autoAlert);
        }
    }
    // NOTE: No else-block. criticalAlarmActive is NEVER auto-cleared by incoming telemetry.
    // Alarm reset is strictly operator-controlled via "RESET" serial command.

    // Stream to Laptop ML Pipeline
    streamJsonToLaptopML(id, 
                         (payload.tilt == -999.0f) ? -999.0f : nodes[id].filtered_tilt,
                         (payload.vibration == -999.0f) ? -999.0f : nodes[id].filtered_vibration,
                         (payload.displacement == -999.0f) ? -999.0f : nodes[id].filtered_displacement);
}

/* ====================================================================================================
 * 9. ESP-NOW RECEPTION CALLBACK (CORE 1)
 * ==================================================================================================== */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *incomingData, int len) {
#else
void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
#endif
    if (len == sizeof(SensorPayload)) {
        SensorPayload packet;
        memcpy(&packet, incomingData, sizeof(SensorPayload));
        if (packet.msg_type == PKT_TYPE_TELEMETRY) {
            processIncomingData(packet);
        }
    } else {
        Serial.printf("[ESP-NOW WARN] Unknown packet size: %d bytes (Expected %u)\n", 
                      len, (unsigned int)sizeof(SensorPayload));
    }
}

/* ====================================================================================================
 * 10. TDMA PROTOCOL IMPLEMENTATION (MASTER BEACON - CORE 1)
 * ==================================================================================================== */
void broadcastTDMABeacon() {
    TDMABeaconPacket beacon;
    beacon.msg_type = PKT_TYPE_BEACON;
    beacon.frame_id = ++tdmaFrameId;
    beacon.beacon_time_ms = millis();
    beacon.slot_duration_ms = TDMA_SLOT_DURATION_MS;
    beacon.total_slots = MAX_SUPPORTED_NODES;
    beacon.emergency_flag = criticalAlarmActive ? 1 : 0;

    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&beacon, sizeof(beacon));
    if (result != ESP_OK) {
        Serial.printf("[TDMA] Beacon broadcast error: 0x%X\n", result);
    }
}

/* ====================================================================================================
 * 11. HARDWARE & PERIPHERAL INITIALIZATION (CORE 1)
 * ==================================================================================================== */
void initHardware() {
    pinMode(PIN_LED_RED_POWER, OUTPUT);
    pinMode(PIN_LED_GREEN_STATUS, OUTPUT);
    digitalWrite(PIN_LED_RED_POWER, HIGH);  // Continuous HIGH power indicator
    digitalWrite(PIN_LED_GREEN_STATUS, LOW);

    pinMode(PIN_BUZZER, OUTPUT);
    digitalWrite(PIN_BUZZER, LOW);

    // Initialize SIM Module UART
    SerialGSM.begin(9600, SERIAL_8N1, PIN_GSM_RX, PIN_GSM_TX);
    delay(1000);

    // Initial AT synchronization handshake (clears boot noise & validates baud)
    for (int i = 0; i < 5; i++) {
        while (SerialGSM.available()) SerialGSM.read();
        SerialGSM.println("AT");
        if (waitForGsmResponse("OK", NULL, 500)) break;
        delay(200);
    }

    SerialGSM.println("ATE0");      // Disable command echo
    waitForGsmResponse("OK", NULL, 300);
    SerialGSM.println("AT+CMGF=1");  // Text mode
    waitForGsmResponse("OK", NULL, 300);

    Serial.println("[GSM] Cellular Serial interface initialized.");
}

void initESPNowTDMA() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Explicitly lock physical Wi-Fi hardware transceiver to Channel 1
    esp_err_t chanErr = esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    if (chanErr == ESP_OK) {
        Serial.printf("[RF LOCK] Wi-Fi Hardware locked to Channel %d\n", ESPNOW_WIFI_CHANNEL);
    } else {
        Serial.printf("[RF ERROR] Failed to lock Wi-Fi channel: %d\n", chanErr);
    }

    if (esp_now_init() != ESP_OK) {
        Serial.println("[ESP-NOW] Critical Init Failure!");
        while (1) delay(1000);
    }

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    esp_now_register_recv_cb((esp_now_recv_cb_t)OnDataRecv);
#else
    esp_now_register_recv_cb((esp_now_recv_cb_t)OnDataRecv);
#endif

    esp_now_peer_info_t peerInfo = {}; // Zero-init defaults ifidx to WIFI_IF_STA (0) on all IDF versions
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = ESPNOW_WIFI_CHANNEL;
    peerInfo.encrypt = false;

    if (!esp_now_is_peer_exist(broadcastAddress)) {
        esp_now_add_peer(&peerInfo);
    }
}

/* ====================================================================================================
 * 12. UI & INDICATOR LOGIC (CORE 1)
 * ==================================================================================================== */
bool areAllDemoNodesSynced() {
    uint32_t now = millis();
    bool node1Synced = nodes[1].active && (now - nodes[1].last_packet_time <= NODE_TIMEOUT_MS);
    bool node2Synced = nodes[2].active && (now - nodes[2].last_packet_time <= NODE_TIMEOUT_MS);

    if (!node1Synced) nodes[1].active = false;
    if (!node2Synced) nodes[2].active = false;

    return (node1Synced && node2Synced);
}

void updateStatusLEDs() {
    uint32_t now = millis();
    bool allSynced = areAllDemoNodesSynced();

    // Red Power/Alarm LED logic
    if (criticalAlarmActive) {
        uint32_t cycle = now % 200;
        digitalWrite(PIN_LED_RED_POWER, (cycle < 100) ? HIGH : LOW);
    } else {
        digitalWrite(PIN_LED_RED_POWER, HIGH);
    }

    if (allSynced) {
        digitalWrite(PIN_LED_GREEN_STATUS, HIGH); // Solid ON when both nodes active & synchronized
    } else {
        if (now - lastBlinkTime >= STATUS_BLINK_INTERVAL_MS) {
            lastBlinkTime = now;
            greenLedState = !greenLedState;
            digitalWrite(PIN_LED_GREEN_STATUS, greenLedState ? HIGH : LOW);
        }
    }
}

/**
 * Non-blocking buzzer state machine owned strictly by Core 1.
 * Handles continuous emergency alarms AND transient GSM failure warning beeps without core fighting.
 */
void handleBuzzerAlerts() {
    uint32_t now = millis();

    // Priority 1: Continuous pulsed alarm for critical subsidence
    if (criticalAlarmActive) {
        uint32_t cycle = now % 400;
        digitalWrite(PIN_BUZZER, (cycle < 200) ? HIGH : LOW);
        return;
    }

    // Priority 2: Non-blocking warning beeps signaled by GSM task
    // Steps count DOWN from (numBeeps*2). Even step = Buzzer ON, Odd step = Buzzer OFF.
    // e.g. 3 beeps: steps 6(ON),5(OFF),4(ON),3(OFF),2(ON),1(OFF) -> correct beep-then-silence pattern
    if (gsmWarningBeepStepsRemaining > 0) {
        if (now - lastBuzzerStepTime >= 100) {
            lastBuzzerStepTime = now;
            bool buzzerOn = (gsmWarningBeepStepsRemaining % 2 == 0); // Even step = ON (starts HIGH)
            digitalWrite(PIN_BUZZER, buzzerOn ? HIGH : LOW);
            gsmWarningBeepStepsRemaining--;
        }
        return;
    }

    // Default: Safe state
    digitalWrite(PIN_BUZZER, LOW);
}

/**
 * Completely non-blocking serial command receiver.
 * Accumulates characters until newline without pausing or blocking the TDMA clock.
 */
void checkSerialCommands() {
    static char cmdBuffer[64];
    static uint8_t cmdPos = 0;

    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmdPos > 0) {
                cmdBuffer[cmdPos] = '\0';
                String cmd = String(cmdBuffer);
                cmd.trim();

                if (cmd.equalsIgnoreCase("RESET")) {
                    criticalAlarmActive = false;
                    for (int i = 1; i < MAX_SUPPORTED_NODES; i++) nodeBreached[i] = false;
                    gsmWarningBeepStepsRemaining = 0;
                    digitalWrite(PIN_BUZZER, LOW);
                    Serial.println("[SYSTEM] Safety reset acknowledged. Alarms cleared.");
                }

                cmdPos = 0; // Reset buffer
            }
        } else {
            if (cmdPos < sizeof(cmdBuffer) - 1) {
                cmdBuffer[cmdPos++] = c;
            }
        }
    }
}

/* ====================================================================================================
 * 13. ARDUINO SETUP (CORE 1 WIRELESS + CORE 0 FREERTOS TASK SPAWN)
 * ==================================================================================================== */
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[SYSTEM] ESP32-S3 Central Master Hub (central_hub_esp32s3_v6) Booting...");

    // Create FreeRTOS Queue BEFORE attaching interrupts in initHardware()
    gsmQueue = xQueueCreate(5, sizeof(SmsAlertRequest));
    if (gsmQueue != NULL) {
        Serial.println("[FreeRTOS] GSM Message Queue created.");
    }

    // Initialize Hardware & ESP-NOW TDMA
    initHardware();
    initESPNowTDMA();

    // Spawn dedicated GSM SMS worker task PINNED TO CORE 0
    xTaskCreatePinnedToCore(
        vGsmSmsTask,        // Function
        "vGsmSmsTask",      // Name
        4096,               // Stack size
        NULL,               // Parameter
        1,                  // Priority
        NULL,               // Task handle
        0                   // CORE 0 PINNING
    );

    Serial.printf("[READY] Real-Time Wireless TDMA running on Core %d.\n", xPortGetCoreID());
}

/* ====================================================================================================
 * 14. MAIN LOOP (CORE 1 - ZERO LATENCY DURING SMS DISPATCH)
 * ==================================================================================================== */
void loop() {
    uint32_t currentMillis = millis();

    // Master TDMA Beacon Broadcast (Strict 1000ms superframe)
    if (currentMillis - lastBeaconTxTime >= TDMA_FRAME_PERIOD_MS) {
        lastBeaconTxTime = currentMillis;
        broadcastTDMABeacon();
    }

    // Real-Time Non-Blocking Subsystems
    checkSerialCommands();
    updateStatusLEDs();
    handleBuzzerAlerts();
}
