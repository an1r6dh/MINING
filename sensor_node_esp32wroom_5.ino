/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
 * FIRMWARE: Sensor Field Node 2 (sensor_node_esp32wroom_5.ino)
 * TARGET: ESP-WROOM-32 Development Board (Arduino IDE Compilable)
 * ====================================================================================================
 * 
 * SENSOR HARDWARE FOR NODE 2:
 * - MPU-6050 6-DOF IMU (I2C: SDA=GPIO 21, SCL=GPIO 22)
 *   - Live Tilt Angle: Continuous degrees from complementary filter
 *   - Live Dynamic Vibration: Dynamic acceleration magnitude (|total_g - 1.0g|)
 *   - Displacement: Fixed 0.0 mm (Ultrasonic sensor is physically wired to Node 1)
 * - Auto-Fallback: Digital Tilt (GPIO 4) & Shock (GPIO 16) if MPU6050 is not attached
 * - TDMA Slot: Slot 2 (160ms to 240ms)
 * - Direct USB Telemetry: Autonomous 500ms broadcast if plugged directly into laptop without Hub
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
#define NODE_ID                 2    // NODE 2 (Assigned Slot 2: 160ms to 240ms)
#define ESPNOW_WIFI_CHANNEL     1    // Must strictly match Central Hub channel (Locked)

// I2C Pins for MPU-6050 (Standard ESP32)
#define PIN_I2C_SDA             21   // Standard ESP32 I2C SDA
#define PIN_I2C_SCL             22   // Standard ESP32 I2C SCL

// Digital Sensor Fallback Pins (ESP-WROOM-32)
#define PIN_DIGITAL_TILT        4    // Digital Tilt Sensor DO Pin (e.g., SW-520D)
#define PIN_DIGITAL_VIBRATION   16   // Digital Vibration Sensor (e.g., SW-420)
#define PIN_STATUS_LED          2    // Onboard Status LED

// Active trigger state for digital comparator modules
#define SENSOR_TRIGGER_LEVEL    LOW  

// Protocol Message Identifiers
#define PKT_TYPE_BEACON         0x01
#define PKT_TYPE_TELEMETRY      0x02

// Central Hub ESP-NOW Broadcast Address
uint8_t hubBroadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

/* ====================================================================================================
 * 2. EXACT PACKED BINARY PAYLOAD STRUCTS
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
    float    tilt;          // Tilt angle in degrees (MPU6050 getAngleX)
    float    vibration;     // Dynamic acceleration in g
    float    displacement;  // Fixed 0.0 mm on Node 2 (Ultrasonic on Node 1)
};

/* ====================================================================================================
 * 3. GLOBAL STATE & SENSOR DRIVERS
 * ==================================================================================================== */
MPU6050 mpu(Wire);
bool mpuAvailable = false;

volatile bool beaconReceived = false;
volatile uint32_t syncLocalTimeMs = 0;
volatile uint16_t currentSlotDurationMs = 80;
bool slotTransmittedForFrame = false;

// Pre-cached sensor telemetry
float cachedTilt = 0.0f;
float cachedVib = 0.0f;

// Autonomous direct USB telemetry timer (for standalone USB connection without Hub beacons)
uint32_t lastDirectUsbTx = 0;

// Digital fallback vibration latch
volatile bool vibrationLatched = false;

void IRAM_ATTR vibrationISR() {
    vibrationLatched = true;
}

/* ====================================================================================================
 * 4. ESP-NOW RECEPTION CALLBACK (TDMA BEACON SYNCHRONIZATION)
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
            slotTransmittedForFrame = false;
        }
    }
}

/* ====================================================================================================
 * 5. TELEMETRY TRANSMISSION (EXECUTED STRICTLY WITHIN ASSIGNED TDMA SLOT 2)
 * ==================================================================================================== */
void transmitTelemetry() {
    SensorPayload payload;
    payload.msg_type = PKT_TYPE_TELEMETRY;
    payload.node_id = NODE_ID;
    payload.tilt = cachedTilt;
    payload.vibration = cachedVib;
    payload.displacement = 0.0f; // Displacement is on Node 1 (Offline)

    esp_err_t result = esp_now_send(hubBroadcastAddress, (uint8_t *)&payload, sizeof(payload));
    if (result == ESP_OK) {
        if (mpuAvailable) {
            Serial.printf("[NODE %d TX SLOT 2] MPU6050 Telemetry: Tilt=%+05.2f deg | Vib=%04.2f g | Disp=0.0mm\n",
                          NODE_ID, payload.tilt, payload.vibration);
        } else {
            Serial.printf("[NODE %d TX SLOT 2] Digital Telemetry: Tilt=%03.1f (%s) | Vib=%03.1f (%s) | Disp=0.0mm\n",
                          NODE_ID,
                          payload.tilt, (payload.tilt > 0.0f ? "TILT TRIGGERED" : "UPRIGHT"),
                          payload.vibration, (payload.vibration > 0.0f ? "LATCHED VIB" : "CALM"));
        }

        digitalWrite(PIN_STATUS_LED, HIGH);
        delayMicroseconds(5000);
        digitalWrite(PIN_STATUS_LED, LOW);
    } else {
        Serial.printf("[NODE %d TX ERROR] ESP-NOW send failed: %d\n", NODE_ID, result);
    }
}

/* ====================================================================================================
 * 6. SETUP
 * ==================================================================================================== */
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.printf("\n[NODE %d] Initializing Sensor Field Node (sensor_node_esp32wroom_5)...\n", NODE_ID);

    // Initialize Status LED
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    // 1. Initialize I2C Bus and check for MPU-6050
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Wire.beginTransmission(0x68);
    byte i2cError = Wire.endTransmission();
    if (i2cError == 0) {
        Serial.println("[SENSOR] MPU-6050 detected at I2C address 0x68!");
        mpu.begin();
        Serial.println("[SENSOR] Calculating gyro offsets (keep sensor steady)...");
        mpu.calcGyroOffsets(true);
        mpuAvailable = true;
        Serial.println("[SENSOR] MPU-6050 calibration complete.");
    } else {
        // Probe alternate I2C address 0x69
        Wire.beginTransmission(0x69);
        i2cError = Wire.endTransmission();
        if (i2cError == 0) {
            Serial.println("[SENSOR] MPU-6050 detected at I2C address 0x69!");
            mpu.begin();
            mpu.calcGyroOffsets(true);
            mpuAvailable = true;
            Serial.println("[SENSOR] MPU-6050 calibration complete.");
        } else {
            Serial.println("[SENSOR NOTICE] MPU-6050 not responding on I2C. Falling back to Digital Pins (4 & 16)...");
            mpuAvailable = false;
            pinMode(PIN_DIGITAL_TILT, INPUT_PULLUP);
            pinMode(PIN_DIGITAL_VIBRATION, INPUT_PULLUP);
            attachInterrupt(digitalPinToInterrupt(PIN_DIGITAL_VIBRATION), vibrationISR, FALLING);
        }
    }

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
 * 7. MAIN LOOP: CONTINUOUS MPU SENSING & STRICT TDMA DISPATCH
 * ==================================================================================================== */
void loop() {
    uint32_t now = millis();

    // 1. Update active sensor readings continuously
    if (mpuAvailable) {
        mpu.update();
        float ax = mpu.getAccX();
        float ay = mpu.getAccY();
        float az = mpu.getAccZ();
        float t_angle = mpu.getAngleX();

        if (isnan(t_angle) || isnan(ax) || isnan(ay) || isnan(az)) {
            cachedTilt = 0.0f;
            cachedVib  = 0.0f;
        } else {
            cachedTilt = t_angle;
            float total_g = sqrt(ax * ax + ay * ay + az * az);
            cachedVib = fabs(total_g - 1.0f);
        }
    } else {
        // Digital fallback reading
        int tiltState = digitalRead(PIN_DIGITAL_TILT);
        cachedTilt = (tiltState == SENSOR_TRIGGER_LEVEL) ? 1.0f : 0.0f;

        noInterrupts();
        bool currentVib = vibrationLatched;
        vibrationLatched = false;
        interrupts();
        cachedVib = currentVib ? 1.0f : 0.0f;
    }

    // Beacon Expiry & Status LED
    if (beaconReceived && (now - syncLocalTimeMs > 2000)) {
        beaconReceived = false;
    }

    if (!beaconReceived && (now - syncLocalTimeMs > 2000)) {
        uint32_t cycle = now % 400;
        digitalWrite(PIN_STATUS_LED, (cycle < 200) ? HIGH : LOW);

        // Standalone Direct USB Serial Fallback (every 500ms when connected directly to computer USB without Hub beacons)
        if (now - lastDirectUsbTx >= 500) {
            lastDirectUsbTx = now;
            if (mpuAvailable) {
                Serial.printf("[NODE %d TX SLOT 2] MPU6050 Telemetry: Tilt=%+05.2f deg | Vib=%04.2f g | Disp=0.0mm\n",
                              NODE_ID, cachedTilt, cachedVib);
            } else {
                Serial.printf("[NODE %d TX SLOT 2] Digital Telemetry: Tilt=%04.1f (%s) | Vib=%04.1f (%s) | Disp=0.0mm\n",
                              NODE_ID, cachedTilt, (cachedTilt > 0.0f ? "TRIGGERED" : "UPRIGHT"),
                              cachedVib, (cachedVib > 0.0f ? "SHOCK" : "CALM"));
            }
        }
    } else if (!beaconReceived) {
        digitalWrite(PIN_STATUS_LED, LOW);
    }

    // 2. TDMA Slot 2 Transmission (160ms to 240ms after Hub Beacon)
    if (beaconReceived && !slotTransmittedForFrame) {
        uint32_t currentMs = millis();
        uint32_t elapsedSinceBeacon = (currentMs >= syncLocalTimeMs) ? (currentMs - syncLocalTimeMs) : 0;
        uint32_t slot2Start = 2 * (uint32_t)currentSlotDurationMs; // 160ms
        uint32_t slot2End   = 3 * (uint32_t)currentSlotDurationMs; // 240ms

        if (elapsedSinceBeacon >= slot2Start && elapsedSinceBeacon < slot2End) {
            transmitTelemetry();
            slotTransmittedForFrame = true;
        } else if (elapsedSinceBeacon >= slot2End) {
            // Guard: Missed window - skip to protect neighboring TDMA slots
            slotTransmittedForFrame = true;
        }
    }
}
