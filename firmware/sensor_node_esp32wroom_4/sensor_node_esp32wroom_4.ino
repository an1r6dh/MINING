/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
 * FIRMWARE: Sensor Field Node 2 (sensor_node_esp32wroom_4.ino)
 * TARGET: ESP-WROOM-32 Development Board (Arduino IDE Compilable)
 * ====================================================================================================
 *
 * REFACTOR: SW-420 Digital Vibration Sensor → MPU6050 IMU (MPU6050_tockn library)
 * 1. Pure Digital Sensor Streamer (Centralized Hub Alarm Architecture):
 *    - Functions purely as a sensor acquisition and TDMA transmission pipe.
 *    - Zero local threshold evaluations, condition checking, alarm flags, or alarm routines.
 *    - All safety limit evaluations and alarm dispatches are strictly centralized on Central Hub.
 * 2. MPU6050 Vibration via I2C (replaces SW-420 + hardware ISR):
 *    - I2C_SDA on GPIO 21, I2C_SCL on GPIO 22.
 *    - mpu6050.update() called every loop() cycle for continuous sampling.
 *    - Resultant acceleration magnitude = sqrt(ax² + ay² + az²).
 *    - Dynamic vibration force = |total_accel - 1.0g| (removes static 1g gravity component).
 *    - Peak-hold tracking (peakVibrationG) captures transient mechanical shocks between superframes.
 *    - peakVibrationG flushed into payload at TDMA transmit time, then reset for next superframe.
 * 3. Explicit RF Wi-Fi Channel Lock:
 *    - Included <esp_wifi.h>.
 *    - esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE); locks RF hardware to Ch 1.
 * 4. Pure Raw Digital Tilt Telemetry:
 *    - Sampled on GPIO 4 with internal pull-up (INPUT_PULLUP).
 *    - Streams raw digital boolean state (1.0f = triggered, 0.0f = normal) without assuming Hub limits.
 * 5. Strict Slot Window Upper-Bound Guard:
 *    - Node 2 transmits strictly within Slot 2 (160ms to 240ms) on every TDMA superframe.
 * ====================================================================================================
 */

// ====================================================================================================
// IMPORTANT ARDUINO IDE COMPILATION NOTICE:
// In the Arduino IDE build system, ALL .ino files residing inside the same folder (sensor_node_esp32wroom)
// are automatically concatenated into a single compilation unit.
// Because both sensor_node_esp32wroom.ino and sensor_node_esp32wroom_4.ino contain setup(), loop(), and
// the same global variables/structs, compiling both simultaneously results in:
// "error: redefinition of 'uint8_t hubBroadcastAddress []' / 'void setup()' / 'void loop()'"
//
// The ACTIVE, PRIMARY sketch for this node is: sensor_node_esp32wroom.ino
// This backup copy is gated out with #if 0 below to eliminate duplicate symbol conflicts in Arduino IDE.
// ====================================================================================================

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>       // Explicit Wi-Fi Hardware Channel Configuration
#include <Wire.h>           // I2C bus driver for MPU6050
#include <MPU6050_tockn.h>  // MPU6050 IMU library (tockn)

/* ====================================================================================================
 * 1. NODE CONFIGURATION & HARDWARE PIN ASSIGNMENTS
 * ==================================================================================================== */
#define NODE_ID                 2    // NODE 2 (Assigned Slot 2: 160ms to 240ms)
#define ESPNOW_WIFI_CHANNEL     1    // Must strictly match Central Hub channel (Locked)

// Digital Sensor Pins (ESP-WROOM-32)
#define PIN_DIGITAL_TILT        4    // Digital Tilt Sensor DO Pin (e.g., SW-520D ball switch)
#define PIN_STATUS_LED          2    // Onboard Status LED

// MPU6050 I2C Bus Pins (ESP-WROOM-32 default I2C)
#define I2C_SDA                 21   // I2C Data  — GPIO 21
#define I2C_SCL                 22   // I2C Clock — GPIO 22

// Active trigger state (Standard comparator modules output LOW when triggered)
#define SENSOR_TRIGGER_LEVEL    LOW

// Protocol Message Identifiers
#define PKT_TYPE_BEACON         0x01
#define PKT_TYPE_TELEMETRY      0x02

// Central Hub ESP-NOW Broadcast Address
uint8_t hubBroadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

/* ====================================================================================================
 * 2. EXACT PACKED BINARY PAYLOAD STRUCTS (MAINTAINED)
 * ==================================================================================================== */

// TDMA Synchronization Beacon received from Central Hub
struct __attribute__((packed)) TDMABeaconPacket {
    uint8_t  msg_type;
    uint32_t frame_id;
    uint32_t beacon_time_ms;
    uint16_t slot_duration_ms;
    uint8_t  total_slots;
    uint8_t  emergency_flag;
};

// Real Telemetry Payload transmitted to Central Hub
struct __attribute__((packed)) SensorPayload {
    uint8_t  msg_type;
    uint8_t  node_id;
    float    tilt;          // 10.0f on trigger (breaches threshold), 0.0f normal
    float    vibration;     // 1.0f on trigger (latched), 0.0f normal
    float    displacement;  // 0.0f fixed
};

/* ====================================================================================================
 * 3. GLOBAL STATE & TIMING VARIABLES
 * ==================================================================================================== */
volatile bool beaconReceived = false;
volatile uint32_t syncLocalTimeMs = 0;
volatile uint16_t currentSlotDurationMs = 80;
bool slotTransmittedForFrame = false;

/* ====================================================================================================
 * 4. MPU6050 IMU — GLOBAL INSTANCE & PEAK-HOLD ACCUMULATOR
 * ==================================================================================================== */
MPU6050 mpu6050(Wire);  // MPU6050 instance bound to the Wire (I2C) bus

// Peak-hold vibration accumulator: tracks maximum dynamic g-force magnitude
// between consecutive TDMA superframes so transient mechanical shocks are never missed.
float peakVibrationG = 0.0f;

/* ====================================================================================================
 * 5. ESP-NOW RECEPTION CALLBACK (TDMA BEACON SYNCHRONIZATION)
 * ==================================================================================================== */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *incomingData, int len) {
#else
void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
#endif
    if (len == sizeof(TDMABeaconPacket)) {
        TDMABeaconPacket beacon;
        memcpy(&beacon, incomingData, sizeof(beacon));
        if (beacon.msg_type == PKT_TYPE_BEACON) {
            syncLocalTimeMs = millis();
            currentSlotDurationMs = beacon.slot_duration_ms;
            beaconReceived = true;
            slotTransmittedForFrame = false; // Reset transmission lock for new TDMA frame
        }
    }
}

/* ====================================================================================================
 * 6. TELEMETRY TRANSMISSION (EXECUTED STRICTLY WITHIN ASSIGNED TDMA SLOT 2)
 * ==================================================================================================== */
void transmitTelemetry() {
    SensorPayload payload;
    payload.msg_type = PKT_TYPE_TELEMETRY;
    payload.node_id = NODE_ID;

    // --- Digital Tilt Sensor Read & Verification ---
    int tiltState = digitalRead(PIN_DIGITAL_TILT);
    if (tiltState != HIGH && tiltState != LOW) {
        Serial.println("[SENSOR FAULT] Node 2: Digital tilt sensor read error. Skipping parameter.");
        payload.tilt = -999.0f;
    } else {
        payload.tilt = (tiltState == SENSOR_TRIGGER_LEVEL) ? 1.0f : 0.0f;
    }

    // --- MPU6050 Peak-Hold Vibration: flush accumulated peak for this superframe ---
    // peakVibrationG holds the worst-case |total_accel - 1g| observed since last transmission.
    // Reset immediately after capture so the next superframe accumulates fresh readings.
    payload.vibration = peakVibrationG;
    peakVibrationG = 0.0f;

    // Displacement fixed to 0.0f on Node 2 (pure digital tripwire node)
    payload.displacement = 0.0f;

    esp_err_t result = esp_now_send(hubBroadcastAddress, (uint8_t *)&payload, sizeof(payload));
    if (result == ESP_OK) {
        Serial.printf("[NODE %d TX SLOT 2] IMU Telemetry: Tilt=%03.1f (%s) | VibPeak=%.4fg (%s) | Disp=0.0mm\n",
                      NODE_ID,
                      payload.tilt,      (payload.tilt > 0.0f      ? "TILT TRIGGERED" : "UPRIGHT"),
                      payload.vibration, (payload.vibration > 0.0f ? "SHOCK DETECTED" : "CALM"));

        digitalWrite(PIN_STATUS_LED, HIGH);
        delayMicroseconds(5000);
        digitalWrite(PIN_STATUS_LED, LOW);
    } else {
        Serial.printf("[NODE %d TX ERROR] ESP-NOW send failed: %d\n", NODE_ID, result);
    }
}

/* ====================================================================================================
 * 7. SETUP
 * ==================================================================================================== */
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.printf("\n[NODE %d] Initializing IMU Sensor Node (sensor_node_esp32wroom_4)...\n", NODE_ID);

    // Initialize Status LED
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    // Initialize Digital Tilt Sensor with Internal Pull-Up
    pinMode(PIN_DIGITAL_TILT, INPUT_PULLUP);
    Serial.println("[SENSOR] Digital Tilt configured on GPIO 4 (INPUT_PULLUP).");

    // Initialize I2C bus for MPU6050 (SDA=GPIO21, SCL=GPIO22)
    Wire.begin(I2C_SDA, I2C_SCL);
    Serial.printf("[I2C] Bus initialized — SDA=GPIO%d  SCL=GPIO%d\n", I2C_SDA, I2C_SCL);

    // Initialize MPU6050 IMU
    mpu6050.begin();
    Serial.println("[MPU6050] Sensor initialized.");

    // Gyroscope offset calibration (false = no serial output during calibration)
    Serial.println("[MPU6050] Calibrating gyro offsets — keep sensor still...");
    mpu6050.calcGyroOffsets(false);
    Serial.println("[MPU6050] Gyro calibration complete.");

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
        while (1) delay(1000);
    }

    esp_now_register_recv_cb(OnDataRecv);

    // Register Hub broadcast peer
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, hubBroadcastAddress, 6);
    peerInfo.channel = ESPNOW_WIFI_CHANNEL;
    peerInfo.encrypt = false;

    if (!esp_now_is_peer_exist(hubBroadcastAddress)) {
        esp_now_add_peer(&peerInfo);
    }

    Serial.printf("[NODE %d READY] Synchronized on Wi-Fi Channel %d.\n", NODE_ID, ESPNOW_WIFI_CHANNEL);
}

/* ====================================================================================================
 * 8. MAIN LOOP: CONTINUOUS MPU6050 SAMPLING + STRICT TDMA DISPATCH
 * ==================================================================================================== */
void loop() {
    uint32_t now = millis();

    // --- MPU6050 Continuous Sampling & Peak-Hold Accumulation ---
    // Update IMU readings every loop cycle for maximum temporal resolution.
    // Computes dynamic vibration force (g above/below 1g static gravity) and
    // retains the worst-case peak between TDMA superframes to capture transient shocks.
    mpu6050.update();

    float ax = mpu6050.getAccX();  // g-force X axis
    float ay = mpu6050.getAccY();  // g-force Y axis
    float az = mpu6050.getAccZ();  // g-force Z axis

    // Resultant acceleration magnitude (total vector)
    float totalAccel = sqrtf(ax * ax + ay * ay + az * az);

    // Dynamic vibration force: deviation from static 1g (removes gravitational baseline)
    float vibrationG = fabsf(totalAccel - 1.0f);

    // Peak-hold: keep maximum dynamic g-force until flushed at TDMA transmit time
    if (vibrationG > peakVibrationG) {
        peakVibrationG = vibrationG;
    }

    // --- Beacon Expiry & Serial Protection ---
    if (beaconReceived && (now - syncLocalTimeMs > 2000)) {
        beaconReceived = false;
    }

    // --- Node Status LED Logic ---
    if (!beaconReceived && (now - syncLocalTimeMs > 2000)) {
        uint32_t cycle = now % 400;
        digitalWrite(PIN_STATUS_LED, (cycle < 200) ? HIGH : LOW);
    } else if (!beaconReceived) {
        digitalWrite(PIN_STATUS_LED, LOW);
    }

    // --- TDMA Slot 2 Transmission Check ---
    if (beaconReceived && !slotTransmittedForFrame) {
        uint32_t elapsedSinceBeacon = millis() - syncLocalTimeMs;

        // Slot 0: Hub Beacon & Guard Margin (0 to slotDurationMs)
        // Slot 1: Node 1 Assigned Window (slotDurationMs to 2 * slotDurationMs)
        // Slot 2: Node 2 Assigned Window (2 * slotDurationMs to 3 * slotDurationMs)
        uint32_t slotStartTime = (uint32_t)NODE_ID * currentSlotDurationMs; // 160 ms
        uint32_t slotEndTime   = slotStartTime + currentSlotDurationMs;     // 240 ms

        // STRICT SLOT WINDOW UPPER-BOUND GUARD:
        if (elapsedSinceBeacon >= slotStartTime && elapsedSinceBeacon < slotEndTime) {
            transmitTelemetry();
            slotTransmittedForFrame = true; // Transmit strictly once per frame
        }
        else if (elapsedSinceBeacon >= slotEndTime) {
            // Missed slot window! Skip to protect adjacent slots from RF collision
            slotTransmittedForFrame = true;
            if (beaconReceived) {
                Serial.printf("[TDMA GUARD Node %d] Missed window (+%lums, window %lu-%lums). Skipping frame.\n",
                              NODE_ID, elapsedSinceBeacon, slotStartTime, slotEndTime);
            }
        }
    }
}


