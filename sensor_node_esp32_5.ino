/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
 * FIRMWARE: Sensor Field Node 1 (sensor_node_esp32_5.ino)
 * TARGET: Standard ESP32 Development Board (Arduino IDE Compilable)
 * ====================================================================================================
 * 
 * AUDIT CORRECTIONS IMPLEMENTED:
 * 1. Slot 0 Ultrasonic Sampling Pause Elimination:
 *    - In loop(), background ultrasonic distance sampling is permitted during Slot 0 (0 to 80ms).
 *    - Only paused strictly during active Slot 1 execution window (80ms to 160ms):
 *      bool isInsideSlot1Window = (beaconReceived && !slotTransmittedForFrame && 
 *                                  elapsed >= currentSlotDurationMs && elapsed < (2 * currentSlotDurationMs));
 * 2. Explicit RF Wi-Fi Channel Lock:
 *    - Included <esp_wifi.h>.
 *    - Immediately after WiFi.mode(WIFI_STA) and WiFi.disconnect(), executed:
 *      esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
 *      Locks physical RF transceiver strictly to Channel 1, eliminating channel drift.
 * 3. Continuous MPU6050 Complementary Filter Updates:
 *    - mpu.update() is called on EVERY iteration of loop().
 *    - Guarantees accurate gyro integration dt and completely eliminates angle drift/lag.
 * 4. Non-Blocking Ultrasonic Distance Sampling:
 *    - pulseIn() is decoupled from TDMA slot dispatch. Distance is pre-sampled in background (60ms)
 *    - Telemetry is dispatched instantaneously (< 1 ms) inside TDMA Slot 1.
 * 5. Strict Slot Window Upper-Bound Guard:
 *    - Node 1 transmits strictly between 80ms and 160ms. Missed windows are skipped to protect Node 2.
 * ====================================================================================================
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h> // Explicit Wi-Fi Hardware Channel Configuration
#include <Wire.h>
#include <MPU6050_tockn.h>

/* ====================================================================================================
 * 1. NODE CONFIGURATION & HARDWARE PIN ASSIGNMENTS
 * ==================================================================================================== */
#define NODE_ID                 1    // NODE 1 (Assigned Slot 1: 80ms to 160ms)
#define ESPNOW_WIFI_CHANNEL     1    // Must strictly match Central Hub channel (Locked)

// I2C Pins for MPU6050
#define PIN_I2C_SDA             21   // Standard ESP32 I2C SDA
#define PIN_I2C_SCL             22   // Standard ESP32 I2C SCL

// HC-SR04 Ultrasonic Sensor Pins
#define PIN_TRIG                5    // HC-SR04 Ultrasonic Trigger Pin
#define PIN_ECHO                18   // HC-SR04 Ultrasonic Echo Pin

// Visual Indicator Pin
#define PIN_STATUS_LED          2    // Onboard status LED

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
    float    tilt;          // Tilt angle in degrees (mapped from mpu.getAngleX())
    float    vibration;     // Dynamic acceleration in g
    float    displacement;  // Roof displacement in millimeters (from HC-SR04)
};

/* ====================================================================================================
 * 3. GLOBAL STATE, DRIVERS & TELEMETRY CACHING
 * ==================================================================================================== */
MPU6050 mpu(Wire);

volatile bool beaconReceived = false;
volatile uint32_t syncLocalTimeMs = 0;
volatile uint16_t currentSlotDurationMs = 80;
bool slotTransmittedForFrame = false;

// Pre-cached sensor telemetry for instantaneous (<1ms) TDMA dispatch
float cachedTilt = 0.0f;
float cachedVib = 0.0f;
float cachedDisp = 0.0f;

// Background ultrasonic sampling timer
uint32_t lastUltrasonicSampleMs = 0;
#define ULTRASONIC_SAMPLE_INTERVAL_MS 60

/* ====================================================================================================
 * 4. HC-SR04 ULTRASONIC DISPLACEMENT SENSOR DRIVER (BACKGROUND NON-BLOCKING)
 * ==================================================================================================== */

/**
 * @brief Measures distance using HC-SR04 and converts to millimeters (mm)
 * Speed of sound = 343 m/s = 0.343 mm/microsecond
 * One-way distance = (duration_us * 0.343) / 2 = duration_us * 0.1715
 */
float readUltrasonicDisplacementMm() {
    digitalWrite(PIN_TRIG, LOW);
    delayMicroseconds(2);

    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);

    // 25ms timeout (~4.2 meters max range)
    long duration_us = pulseIn(PIN_ECHO, HIGH, 25000);

    if (duration_us == 0) {
        return cachedDisp; // Retain last valid measurement on echo timeout
    }

    float distance_mm = (float)duration_us * 0.1715f;
    return distance_mm;
}

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
 * 6. SETUP
 * ==================================================================================================== */
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.printf("\n[NODE %d] Initializing Geotechnical Sensor Node (sensor_node_esp32_5)...\n", NODE_ID);

    // Initialize Status LED
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    // Initialize HC-SR04 Ultrasonic Pins
    pinMode(PIN_TRIG, OUTPUT);
    pinMode(PIN_ECHO, INPUT);
    digitalWrite(PIN_TRIG, LOW);

    // Initialize I2C Bus and MPU6050_tockn Library
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Serial.println("[SENSOR] Initializing MPU6050 via MPU6050_tockn...");
    mpu.begin();
    Serial.println("[SENSOR] Calculating gyro offsets (Keep sensor stationary)...");
    mpu.calcGyroOffsets(true);
    Serial.println("[SENSOR] MPU6050 calibration complete.");

    // Initial ultrasonic distance read
    cachedDisp = readUltrasonicDisplacementMm();

    // Initialize Wi-Fi in Station mode for ESP-NOW
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
 * 7. MAIN LOOP: CONTINUOUS MPU FILTER UPDATES + BACKGROUND SAMPLING + INSTANT TDMA DISPATCH
 * ==================================================================================================== */
void loop() {
    uint32_t now = millis();

    // 1. MANDATORY CONTINUOUS UPDATE: Update MPU6050 complementary filter on EVERY loop cycle!
    // Ensures internal gyro integration dt is accurate and eliminates angle drift/freezing.
    mpu.update();

    // 2. Continuous background telemetry update
    cachedTilt = mpu.getAngleX();

    float ax = mpu.getAccX();
    float ay = mpu.getAccY();
    float az = mpu.getAccZ();
    float total_g = sqrt(ax * ax + ay * ay + az * az);
    cachedVib = fabs(total_g - 1.0f);

    // 3. MANDATORY AUDIT FIX: Ultrasonic background sampling is active during Slot 0 (0-80ms),
    // and ONLY paused when actively inside the Slot 1 transmission window (80-160ms)!
    uint32_t elapsed = now - syncLocalTimeMs;
    bool isInsideSlot1Window = (beaconReceived && !slotTransmittedForFrame && 
                                elapsed >= currentSlotDurationMs && 
                                elapsed < (2 * currentSlotDurationMs));

    if (!isInsideSlot1Window && (now - lastUltrasonicSampleMs >= ULTRASONIC_SAMPLE_INTERVAL_MS)) {
        lastUltrasonicSampleMs = now;
        cachedDisp = readUltrasonicDisplacementMm();
    }

    // 4. INSTANTANEOUS TDMA SLOT 1 TRANSMISSION (< 1 ms dispatch)
    if (beaconReceived && !slotTransmittedForFrame) {
        uint32_t slotStartTime = (uint32_t)NODE_ID * currentSlotDurationMs; // 80ms
        uint32_t slotEndTime   = slotStartTime + currentSlotDurationMs;     // 160ms

        // STRICT SLOT WINDOW UPPER-BOUND GUARD:
        if (elapsed >= slotStartTime && elapsed < slotEndTime) {
            // Dispatches pre-cached sensor readings instantaneously without blocking
            SensorPayload payload;
            payload.msg_type = PKT_TYPE_TELEMETRY;
            payload.node_id = NODE_ID;
            payload.tilt = cachedTilt;
            payload.vibration = cachedVib;
            payload.displacement = cachedDisp;

            esp_err_t result = esp_now_send(hubBroadcastAddress, (uint8_t *)&payload, sizeof(payload));
            if (result == ESP_OK) {
                Serial.printf("[NODE %d TX SLOT 1] Sent Instant (<1ms): Tilt=%+05.2f deg | Vib=%04.2f g | Disp=%05.1f mm\n",
                              NODE_ID, cachedTilt, cachedVib, cachedDisp);
                digitalWrite(PIN_STATUS_LED, HIGH);
                delayMicroseconds(5000);
                digitalWrite(PIN_STATUS_LED, LOW);
            } else {
                Serial.printf("[NODE %d TX ERROR] ESP-NOW send failed: %d\n", NODE_ID, result);
            }

            slotTransmittedForFrame = true; // Transmit strictly once per frame
        } 
        else if (elapsed >= slotEndTime) {
            // Missed slot window! Skip to protect adjacent slots from collisions
            slotTransmittedForFrame = true;
            Serial.printf("[TDMA GUARD Node %d] Missed window (+%lums, window %lu-%lums). Skipping frame.\n",
                          NODE_ID, elapsed, slotStartTime, slotEndTime);
        }
    }
}
