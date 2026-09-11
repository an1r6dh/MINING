/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING & EARLY WARNING (SIH 26025)
 * FIRMWARE: ESP32-S3 Central Master Hub (central_hub_esp32s3_5.ino)
 * TARGET: ESP32-S3 Development Board (Arduino IDE Compilable)
 * ====================================================================================================
 * 
 * AUDIT CORRECTIONS IMPLEMENTED:
 * 1. Queue Initialization Race Fix (CRITICAL):
 *    - In setup(), gsmQueue = xQueueCreate(...) is now called BEFORE initHardware().
 *    - Prevents missed SMS alert requests if GPIO 10 transitions LOW during boot interrupt attach.
 * 2. 16x2 LCD Character Truncation Fix:
 *    - In updateLCDDisplay(), formatted line0 with %+04.1f:
 *      snprintf(line0, sizeof(line0), "N%d T:%+04.1f V:%.2f", ...)
 *    - Strictly formats to 16 characters ("N1 T:+0.0 V:0.00"), preventing 2nd decimal truncation.
 * 3. Explicit RF Wi-Fi Channel Lock:
 *    - Included <esp_wifi.h>.
 *    - esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE); locks RF hardware to Ch 1.
 * 4. FreeRTOS Dual-Core Architecture Retained:
 *    - Core 1: Non-blocking TDMA master beacon (1000ms), ESP-NOW telemetry processing, LCD & LEDs.
 *    - Core 0: vGsmSmsTask pinned via xTaskCreatePinnedToCore() for asynchronous SIM300L SMS alerts.
 * 5. Geotechnical Kalman State Isolation & Digital Override Retained:
 *    - Node 1: Processed through dedicated 1D Kalman filters (kfTilt, kfVibration, kfDisplacement).
 *    - Node 2: Direct digital override (bypasses Kalman filters for instantaneous alarm tripping).
 * ====================================================================================================
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h> // Explicit Wi-Fi Hardware Channel Configuration
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

/* ====================================================================================================
 * 1. HARDWARE PIN DEFINITIONS (ESP32-S3 SPECIFIC)
 * ==================================================================================================== */
#define PIN_LED_RED_POWER       1    // Power LED (Continuously driven HIGH in setup)
#define PIN_LED_GREEN_STATUS    2    // Status LED (Solid = Nodes 1 & 2 synced; Rapid blink = timeout)
#define PIN_BUZZER              4    // High-pitch alarm buzzer (Active buzzer or PWM drive)

#define PIN_SERVER_INTERRUPT    10   // AI Server Hardware Interrupt (Active LOW with internal pull-up)

#define PIN_GSM_RX              18   // ESP32-S3 RX1 -> connected to SIM300L TX
#define PIN_GSM_TX              17   // ESP32-S3 TX1 -> connected to SIM300L RX

#define PIN_I2C_SDA             8    // ESP32-S3 hardware I2C SDA
#define PIN_I2C_SCL             9    // ESP32-S3 hardware I2C SCL
#define LCD_I2C_ADDR            0x27 // Default PCF8574 backpack address (change to 0x3F if required)
#define LCD_COLS                16
#define LCD_ROWS                2

/* ====================================================================================================
 * 2. SYSTEM CONFIGURATION & SAFETY THRESHOLDS
 * ==================================================================================================== */
#define ESPNOW_WIFI_CHANNEL     1               // Channel for ESP-NOW TDMA cluster (RF Locked)
#define EMERGENCY_PHONE_NUMBER  "+919876543210" // Destination phone number for SMS alerts

#define CRITICAL_DISPLACEMENT_THRESH_MM 15.0f   // Critical rock mass displacement trigger (mm)
#define CRITICAL_TILT_THRESH_DEG        5.0f    // Critical angular tilt trigger (degrees)
#define NODE_TIMEOUT_MS                 3500    // Timeout threshold before marking a node offline

#define TDMA_FRAME_PERIOD_MS    1000            // Master TDMA superframe period (1000 ms)
#define TDMA_SLOT_DURATION_MS   80              // Transmit window duration per node (80 ms)
#define LCD_REFRESH_INTERVAL_MS 250             // Non-blocking LCD refresh interval
#define LCD_ROTATE_INTERVAL_MS  2000            // Rotate LCD telemetry between Node 1 & 2 every 2s
#define STATUS_BLINK_INTERVAL_MS 150            // Rapid blink rate when any demo node times out
#define SMS_COOLDOWN_MS         60000           // 60s cooldown between automated threshold SMS sends

/* ====================================================================================================
 * 3. EMBEDDED 1D DISCRETE KALMAN FILTER ENGINE
 * ==================================================================================================== */
class DiscreteKalmanFilter1D {
public:
    DiscreteKalmanFilter1D(float process_noise = 0.02f, float measurement_noise = 1.5f, float estimation_error = 1.0f, float initial_value = 0.0f)
        : _q(process_noise), _r(measurement_noise), _p(estimation_error), _x(initial_value), _k(0.0f) {}

    float update(float measurement) {
        _p = _p + _q;
        float denominator = _p + _r;
        _k = (fabs(denominator) > 1e-6f) ? (_p / denominator) : 0.5f;
        _x = _x + _k * (measurement - _x);
        _p = (1.0f - _k) * _p;
        return _x;
    }

    float getState() const { return _x; }
    void setNoiseParameters(float q, float r) { _q = q; _r = r; }

private:
    float _q;
    float _r;
    float _p;
    float _x;
    float _k;
};

/* ====================================================================================================
 * 4. PROTOCOL DATA STRUCTURES (EXACT PACKED BINARY PAYLOADS)
 * ==================================================================================================== */
#define PKT_TYPE_BEACON    0x01
#define PKT_TYPE_TELEMETRY 0x02

struct __attribute__((packed)) TDMABeaconPacket {
    uint8_t  msg_type;           // PKT_TYPE_BEACON
    uint32_t frame_id;          // Monotonically increasing frame counter
    uint32_t beacon_time_ms;    // Master clock timestamp
    uint16_t slot_duration_ms;  // Slot duration (80ms)
    uint8_t  total_slots;       // Number of active slots in frame
    uint8_t  emergency_flag;    // 1 if alarm active, 0 normal
};

struct __attribute__((packed)) SensorPayload {
    uint8_t  msg_type;          // PKT_TYPE_TELEMETRY
    uint8_t  node_id;           // 1 = Node 1, 2 = Node 2
    float    tilt;              // Raw Tilt angle (degrees)
    float    vibration;         // Raw Vibration acceleration (g or m/s^2)
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
LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);
bool lcdAvailable = false;

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
    DiscreteKalmanFilter1D(0.01f, 0.8f), // Node 1 Ultrasonic Displacement
    DiscreteKalmanFilter1D(0.01f, 0.8f)  // Node 2 (Bypassed)
};

FilteredNodeRecord nodes[MAX_SUPPORTED_NODES];

// System State & Non-Blocking Timers
uint32_t tdmaFrameId = 0;
uint32_t lastBeaconTxTime = 0;
uint32_t lastLcdUpdateTime = 0;
uint32_t lastLcdRotateTime = 0;
uint8_t  currentLcdDisplayNode = 1;
uint32_t lastBlinkTime = 0;
bool     greenLedState = false;
uint32_t lastAutoSmsDispatchTime = 0;

volatile bool serverInterruptTriggered = false;
bool smsAlertDispatchedForAi = false;
bool criticalAlarmActive = false;

/* ====================================================================================================
 * 6. FORWARD DECLARATIONS
 * ==================================================================================================== */
void vGsmSmsTask(void *pvParameters);
void sendSMSAlertBlocking(const char* alertMessage);
void triggerSmsAlert(const char* alertMsg);
void IRAM_ATTR handleServerInterruptISR();
void processIncomingData(const SensorPayload& payload);
void broadcastTDMABeacon();
void initHardware();
void initESPNowTDMA();
bool areAllDemoNodesSynced();
void updateStatusLEDs();
void handleBuzzerAlerts();
void lcdPrintPaddedLine(uint8_t row, const char* text);
void updateLCDDisplay();
void checkSerialCommands();

/* ====================================================================================================
 * 7. HARDWARE INTERRUPT SERVICE ROUTINE (ISR) FOR AI SERVER
 * ==================================================================================================== */
void IRAM_ATTR handleServerInterruptISR() {
    serverInterruptTriggered = true;
    if (gsmQueue != NULL) {
        SmsAlertRequest req;
        const char* msg = "EMERGENCY: AI Model predicts mine subsidence event! Evacuate Sector B!";
        strncpy(req.message, msg, sizeof(req.message) - 1);
        req.message[sizeof(req.message) - 1] = '\0';
        
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        xQueueSendFromISR(gsmQueue, &req, &xHigherPriorityTaskWoken);
        if (xHigherPriorityTaskWoken) {
            portYIELD_FROM_ISR();
        }
    }
}

/* ====================================================================================================
 * 8. FREERTOS CORE 0 CELLULAR GSM TASK (NON-BLOCKING BACKGROUND DISPATCH)
 * ==================================================================================================== */
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

void sendSMSAlertBlocking(const char* alertMessage) {
    while (SerialGSM.available()) SerialGSM.read();

    SerialGSM.println("AT+CMGF=1");
    vTaskDelay(pdMS_TO_TICKS(200));

    SerialGSM.print("AT+CMGS=\"");
    SerialGSM.print(EMERGENCY_PHONE_NUMBER);
    SerialGSM.println("\"");

    uint32_t promptTimeout = millis();
    bool promptReceived = false;
    while (millis() - promptTimeout < 5000) {
        if (SerialGSM.available()) {
            if (SerialGSM.read() == '>') {
                promptReceived = true;
                break;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    if (!promptReceived) {
        Serial.println("[GSM ERROR Core 0] Prompt '>' not received from SIM module!");
        return;
    }

    SerialGSM.print(alertMessage);
    vTaskDelay(pdMS_TO_TICKS(100));
    SerialGSM.write(0x1A); // Commit SMS with Ctrl+Z

    uint32_t sendTimeout = millis();
    bool sendSuccess = false;
    String gsmResponse = "";
    while (millis() - sendTimeout < 15000) {
        while (SerialGSM.available()) {
            char c = SerialGSM.read();
            gsmResponse += c;
        }
        if (gsmResponse.indexOf("OK") != -1 || gsmResponse.indexOf("+CMGS:") != -1) {
            sendSuccess = true;
            break;
        }
        if (gsmResponse.indexOf("ERROR") != -1) break;
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    if (sendSuccess) {
        Serial.println("[GSM SUCCESS Core 0] Emergency SMS successfully delivered to cellular network!");
    } else {
        Serial.printf("[GSM ERROR Core 0] SMS transmission failed: %s\n", gsmResponse.c_str());
    }
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
 * 9. TELEMETRY PROCESSING: KALMAN FILTERING (NODE 1) & DIGITAL OVERRIDE (NODE 2)
 * ==================================================================================================== */
void processIncomingData(const SensorPayload& payload) {
    uint8_t id = payload.node_id;
    if (id == 0 || id >= MAX_SUPPORTED_NODES) return;

    nodes[id].active = true;
    nodes[id].last_packet_time = millis();
    nodes[id].raw_tilt = payload.tilt;
    nodes[id].raw_vibration = payload.vibration;
    nodes[id].raw_displacement = payload.displacement;

    if (id == 1) {
        nodes[1].filtered_tilt = kfTilt[1].update(payload.tilt);
        nodes[1].filtered_vibration = kfVibration[1].update(payload.vibration);
        nodes[1].filtered_displacement = kfDisplacement[1].update(payload.displacement);
    } 
    else if (id == 2) {
        nodes[2].filtered_tilt = payload.tilt;                 // 10.0f on trigger (breaches 5.0f limit instantly)
        nodes[2].filtered_vibration = payload.vibration;       // 1.0f on trigger
        nodes[2].filtered_displacement = payload.displacement; // 0.0f
    }
    else {
        nodes[id].filtered_tilt = kfTilt[id].update(payload.tilt);
        nodes[id].filtered_vibration = kfVibration[id].update(payload.vibration);
        nodes[id].filtered_displacement = kfDisplacement[id].update(payload.displacement);
    }

    Serial.printf("[NODE %d %s] Tilt: %+05.2f deg | Vib: %04.2f g | Disp: %05.1f mm\n",
                  id, (id == 2 ? "DIGITAL OVERRIDE" : "KALMAN FILTERED"),
                  nodes[id].filtered_tilt, nodes[id].filtered_vibration, nodes[id].filtered_displacement);

    if (nodes[id].filtered_displacement >= CRITICAL_DISPLACEMENT_THRESH_MM ||
        fabs(nodes[id].filtered_tilt) >= CRITICAL_TILT_THRESH_DEG) {
        criticalAlarmActive = true;
        Serial.printf("[LOCAL ALERT] Node %d breached safety limit! Tilt: %.1f deg, Disp: %.1f mm\n",
                      id, nodes[id].filtered_tilt, nodes[id].filtered_displacement);

        uint32_t now = millis();
        if (now - lastAutoSmsDispatchTime > SMS_COOLDOWN_MS) {
            lastAutoSmsDispatchTime = now;
            char autoAlert[140];
            snprintf(autoAlert, sizeof(autoAlert),
                     "CRITICAL ALERT: Mine Movement at Node %d! Tilt: %.1fdeg, Disp: %.1fmm.",
                     id, nodes[id].filtered_tilt, nodes[id].filtered_displacement);
            triggerSmsAlert(autoAlert);
        }
    } else {
        if (!serverInterruptTriggered) {
            bool anyNodeBreached = false;
            for (int i = 1; i < MAX_SUPPORTED_NODES; i++) {
                if (nodes[i].active &&
                    (nodes[i].filtered_displacement >= CRITICAL_DISPLACEMENT_THRESH_MM ||
                     fabs(nodes[i].filtered_tilt) >= CRITICAL_TILT_THRESH_DEG)) {
                    anyNodeBreached = true;
                    break;
                }
            }
            if (!anyNodeBreached) {
                criticalAlarmActive = false;
            }
        }
    }
}

/* ====================================================================================================
 * 10. ESP-NOW RECEPTION CALLBACK (CORE 1)
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
        Serial.printf("[ESP-NOW WARN] Unknown packet size: %d bytes\n", len);
    }
}

/* ====================================================================================================
 * 11. TDMA PROTOCOL IMPLEMENTATION (MASTER BEACON - CORE 1)
 * ==================================================================================================== */
void broadcastTDMABeacon() {
    TDMABeaconPacket beacon;
    beacon.msg_type = PKT_TYPE_BEACON;
    beacon.frame_id = ++tdmaFrameId;
    beacon.beacon_time_ms = millis();
    beacon.slot_duration_ms = TDMA_SLOT_DURATION_MS;
    beacon.total_slots = MAX_SUPPORTED_NODES;
    beacon.emergency_flag = (criticalAlarmActive || serverInterruptTriggered) ? 1 : 0;

    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&beacon, sizeof(beacon));
    if (result != ESP_OK) {
        Serial.printf("[TDMA] Beacon broadcast error: %d\n", result);
    }
}

/* ====================================================================================================
 * 12. HARDWARE & PERIPHERAL INITIALIZATION (CORE 1)
 * ==================================================================================================== */
void initHardware() {
    pinMode(PIN_LED_RED_POWER, OUTPUT);
    pinMode(PIN_LED_GREEN_STATUS, OUTPUT);
    digitalWrite(PIN_LED_RED_POWER, HIGH);  // RED LED set HIGH continuously to denote system power
    digitalWrite(PIN_LED_GREEN_STATUS, LOW);

    pinMode(PIN_BUZZER, OUTPUT);
    digitalWrite(PIN_BUZZER, LOW);

    pinMode(PIN_SERVER_INTERRUPT, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_SERVER_INTERRUPT), handleServerInterruptISR, FALLING);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Wire.beginTransmission(LCD_I2C_ADDR);
    if (Wire.endTransmission() == 0) {
        lcdAvailable = true;
        lcd.init();
        lcd.backlight();
        lcd.clear();
        lcdPrintPaddedLine(0, "MINE SUBSIDENCE");
        lcdPrintPaddedLine(1, "HUB ON - TDMA CH1");
        Serial.println("[I2C] LCD 16x2 detected and ready.");
    } else {
        lcdAvailable = false;
        Serial.println("[I2C WARN] LCD 16x2 not detected on 0x27.");
    }

    SerialGSM.begin(9600, SERIAL_8N1, PIN_GSM_RX, PIN_GSM_TX);
    delay(1000);
    SerialGSM.println("AT");
    delay(200);
    SerialGSM.println("ATE0");
    delay(200);
    SerialGSM.println("AT+CMGF=1");
}

void initESPNowTDMA() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Explicitly lock physical Wi-Fi hardware to Channel 1
    esp_err_t chanErr = esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    if (chanErr == ESP_OK) {
        Serial.printf("[RF LOCK] Wi-Fi Hardware locked to Channel %d\n", ESPNOW_WIFI_CHANNEL);
    } else {
        Serial.printf("[RF ERROR] Failed to lock Wi-Fi channel: %d\n", chanErr);
    }

    if (esp_now_init() != ESP_OK) {
        Serial.println("[ESP-NOW] Critical Init Failure!");
        if (lcdAvailable) {
            lcd.clear();
            lcdPrintPaddedLine(0, "ESP-NOW ERR!");
        }
        while (1) delay(1000);
    }

    esp_now_register_recv_cb(OnDataRecv);

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = ESPNOW_WIFI_CHANNEL;
    peerInfo.encrypt = false;

    if (!esp_now_is_peer_exist(broadcastAddress)) {
        esp_now_add_peer(&peerInfo);
    }
}

/* ====================================================================================================
 * 13. UI & INDICATOR LOGIC (CORE 1) - ZERO-GHOSTING LCD PRINT ENGINE
 * ==================================================================================================== */

void lcdPrintPaddedLine(uint8_t row, const char* text) {
    if (!lcdAvailable) return;
    char buffer[17];
    snprintf(buffer, sizeof(buffer), "%-16.16s", text);
    lcd.setCursor(0, row);
    lcd.print(buffer);
}

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

    if (allSynced) {
        digitalWrite(PIN_LED_GREEN_STATUS, HIGH); // Solid ON when both nodes active & synced
    } else {
        if (now - lastBlinkTime >= STATUS_BLINK_INTERVAL_MS) {
            lastBlinkTime = now;
            greenLedState = !greenLedState;
            digitalWrite(PIN_LED_GREEN_STATUS, greenLedState ? HIGH : LOW);
        }
    }
}

void handleBuzzerAlerts() {
    if (criticalAlarmActive || serverInterruptTriggered) {
        uint32_t cycle = millis() % 400;
        digitalWrite(PIN_BUZZER, (cycle < 200) ? HIGH : LOW);
    } else {
        digitalWrite(PIN_BUZZER, LOW);
    }
}

void updateLCDDisplay() {
    if (!lcdAvailable) return;

    uint32_t now = millis();

    if (now - lastLcdRotateTime >= LCD_ROTATE_INTERVAL_MS) {
        lastLcdRotateTime = now;
        currentLcdDisplayNode = (currentLcdDisplayNode == 1) ? 2 : 1;
    }

    if (now - lastLcdUpdateTime < LCD_REFRESH_INTERVAL_MS) return;
    lastLcdUpdateTime = now;

    if (serverInterruptTriggered) {
        lcdPrintPaddedLine(0, "! AI PREDICTION !");
        lcdPrintPaddedLine(1, "EVACUATE MINE NOW");
        return;
    }

    if (criticalAlarmActive) {
        char critLine[17];
        snprintf(critLine, sizeof(critLine), "N%d T:%.0f D:%.0fmm", 
                 currentLcdDisplayNode, 
                 nodes[currentLcdDisplayNode].filtered_tilt, 
                 nodes[currentLcdDisplayNode].filtered_displacement);
        lcdPrintPaddedLine(0, "CRITICAL SUBSIDE");
        lcdPrintPaddedLine(1, critLine);
        return;
    }

    char line0[24];
    char line1[24];
    uint8_t n = currentLcdDisplayNode;

    if (nodes[n].active && (now - nodes[n].last_packet_time <= NODE_TIMEOUT_MS)) {
        // MANDATORY AUDIT FIX: Formatted with %+04.1f so line0 is strictly 16 chars ("N1 T:+0.0 V:0.00")
        // Eliminates second decimal truncation on Vibration reading!
        snprintf(line0, sizeof(line0), "N%d T:%+04.1f V:%.2f",
                 n, nodes[n].filtered_tilt, nodes[n].filtered_vibration);

        // Padded status indicator (%-5s) guarantees exact 16-character row
        const char* statusTag = (n == 2) ? "[DIG]" : "[OK] ";
        snprintf(line1, sizeof(line1), "D:%04.1fmm %-5s",
                 nodes[n].filtered_displacement, statusTag);
    } else {
        snprintf(line0, sizeof(line0), "NODE %d: TIMEOUT", n);
        snprintf(line1, sizeof(line1), "FRAME #%lu SYNC", tdmaFrameId);
    }

    lcdPrintPaddedLine(0, line0);
    lcdPrintPaddedLine(1, line1);
}

void checkSerialCommands() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.startsWith("ALERT") || cmd.startsWith("INTERRUPT")) {
            Serial.printf("[AI COMMAND] Emergency alert: %s\n", cmd.c_str());
            serverInterruptTriggered = true;
            triggerSmsAlert("EMERGENCY: AI Model predicts mine subsidence event at Section B! Evacuate!");
        } else if (cmd.equalsIgnoreCase("RESET")) {
            serverInterruptTriggered = false;
            criticalAlarmActive = false;
            smsAlertDispatchedForAi = false;
            digitalWrite(PIN_BUZZER, LOW);
            Serial.println("[SYSTEM] System reset to safe.");
        }
    }
}

/* ====================================================================================================
 * 14. ARDUINO SETUP (CORE 1 WIRELESS + CORE 0 FREERTOS TASK SPAWN)
 * ==================================================================================================== */
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[SYSTEM] ESP32-S3 Central Master Hub (central_hub_esp32s3_5) Booting...");

    // MANDATORY AUDIT FIX: 1. Create FreeRTOS Queue BEFORE attaching interrupts in initHardware()
    // Prevents missed SMS alerts if GPIO 10 transitions LOW during bootup
    gsmQueue = xQueueCreate(5, sizeof(SmsAlertRequest));
    if (gsmQueue != NULL) {
        Serial.println("[FreeRTOS] GSM Message Queue created.");
    }

    // 2. Initialize Hardware & Interrupts safely with valid queue
    initHardware();
    initESPNowTDMA();

    // 3. Spawn dedicated GSM SMS worker task PINNED TO CORE 0
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
 * 15. MAIN LOOP (CORE 1 - ZERO LATENCY DURING SMS DISPATCH)
 * ==================================================================================================== */
void loop() {
    uint32_t currentMillis = millis();

    if (currentMillis - lastBeaconTxTime >= TDMA_FRAME_PERIOD_MS) {
        lastBeaconTxTime = currentMillis;
        broadcastTDMABeacon();
    }

    checkSerialCommands();
    updateStatusLEDs();
    handleBuzzerAlerts();
    updateLCDDisplay();
}
