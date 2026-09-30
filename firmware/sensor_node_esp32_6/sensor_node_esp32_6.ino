/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
 * FIRMWARE: Sensor Field Node 1 (sensor_node_esp32_6.ino)
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
volatile bool slotTransmittedForFrame = false; // Written by ESP-NOW ISR (OnDataRecv) — must be volatile

// Pre-cached sensor telemetry for instantaneous (<1ms) TDMA dispatch
float cachedTilt = 0.0f;
float cachedVib = 0.0f;
float cachedDisp = 0.0f;

// Background ultrasonic sampling timer
uint32_t lastUltrasonicSampleMs = 0;
#define ULTRASONIC_SAMPLE_INTERVAL_MS 100

// --- Interrupt-driven HC-SR04 echo timing ---
// pulseIn() busy-waits and is fatally disrupted by ESP-NOW Wi-Fi radio ISRs.
// Instead, we record echo rise/fall timestamps via a GPIO ISR.
volatile uint32_t echoRiseUs  = 0;
volatile uint32_t echoFallUs  = 0;
volatile bool     echoReady   = false; // true once a complete pulse has been captured
volatile bool     echoWaiting = false; // true between trigger and first rising edge

void IRAM_ATTR echoISR() {
    if (digitalRead(PIN_ECHO) == HIGH) {
        // Rising edge — pulse has started
        echoRiseUs  = micros();
        echoReady   = false;
        echoWaiting = true;
    } else {
        // Falling edge — pulse complete
        if (echoWaiting) {
            echoFallUs  = micros();
            echoReady   = true;
            echoWaiting = false;
        }
    }
}

/* ====================================================================================================
 * 4. HC-SR04 ULTRASONIC DISPLACEMENT SENSOR DRIVER (INTERRUPT-DRIVEN, NON-BLOCKING)
 * ==================================================================================================== */

/**
 * @brief Fires the HC-SR04 trigger pulse. The echo is captured asynchronously by echoISR().
 *        Call this, then poll readCachedUltrasonicMm() after ~38ms for the result.
 *
 * Root-cause fix: pulseIn() busy-waits and is fatally preempted by ESP-NOW Wi-Fi radio ISRs,
 * causing it to always return 0 and log a false SENSOR FAULT. The ISR approach records
 * echo rise/fall timestamps independently of the main execution context.
 */
void triggerUltrasonic() {
    echoReady   = false;
    echoWaiting = false;
    digitalWrite(PIN_TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);
    // echoISR() will fire on the rising/falling edge of PIN_ECHO
}

/**
 * @brief Returns the distance in mm from the last completed echo capture, or -999 on timeout.
 *        Must be called at least ~38ms after triggerUltrasonic().
 */
float readCachedUltrasonicMm() {
    // ATOMIC SNAPSHOT: echoRiseUs and echoFallUs are both written by echoISR().
    // Reading them across two separate statements without disabling interrupts risks
    // a torn read — the ISR could update echoFallUs between the two reads, producing
    // a near-zero or garbage duration_us. Snapshot both atomically.
    noInterrupts();
    bool     ready  = echoReady;
    uint32_t rise   = echoRiseUs;
    uint32_t fall   = echoFallUs;
    interrupts();

    if (!ready) {
        Serial.println("[SENSOR FAULT] Node 1: Ultrasonic echo timeout/disconnected. Skipping displacement parameter.");
        return -999.0f;
    }
    uint32_t duration_us = fall - rise;
    if (duration_us == 0 || duration_us > 38000) {
        Serial.println("[SENSOR FAULT] Node 1: Ultrasonic echo out of range. Skipping displacement parameter.");
        return -999.0f;
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
    Serial.printf("\n[NODE %d] Initializing Geotechnical Sensor Node (sensor_node_esp32_6)...\n", NODE_ID);

    // Initialize Status LED
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    // Initialize HC-SR04 Ultrasonic Pins
    pinMode(PIN_TRIG, OUTPUT);
    pinMode(PIN_ECHO, INPUT);
    digitalWrite(PIN_TRIG, LOW);

    // Attach interrupt-driven echo capture (CHANGE = both rising and falling edges)
    attachInterrupt(digitalPinToInterrupt(PIN_ECHO), echoISR, CHANGE);

    // Initialize I2C Bus and MPU6050_tockn Library
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Serial.println("[SENSOR] Initializing MPU6050 via MPU6050_tockn...");
    mpu.begin();
    Serial.println("[SENSOR] Calculating gyro offsets (Keep sensor stationary)...");
    mpu.calcGyroOffsets(true);
    Serial.println("[SENSOR] MPU6050 calibration complete.");

    // Initial ultrasonic trigger (result captured asynchronously via echoISR)
    triggerUltrasonic();
    delay(50); // Wait for first echo to complete before entering loop
    cachedDisp = readCachedUltrasonicMm();

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

    // Beacon Expiry & Serial Protection
    if (beaconReceived && (now - syncLocalTimeMs > 2000)) {
        beaconReceived = false;
    }

    // Node Status LED Logic
    if (!beaconReceived && (now - syncLocalTimeMs > 2000)) {
        uint32_t cycle = now % 400;
        digitalWrite(PIN_STATUS_LED, (cycle < 200) ? HIGH : LOW);
    } else if (!beaconReceived) {
        digitalWrite(PIN_STATUS_LED, LOW);
    }

    // 1. MANDATORY CONTINUOUS UPDATE: Update MPU6050 complementary filter on EVERY loop cycle!
    // Ensures internal gyro integration dt is accurate and eliminates angle drift/freezing.
    mpu.update();

    // 2. Continuous background telemetry update
    float ax = mpu.getAccX();
    float ay = mpu.getAccY();
    float az = mpu.getAccZ();
    float t_angle = mpu.getAngleX();

    if (isnan(t_angle) || isnan(ax) || isnan(ay) || isnan(az)) {
        Serial.println("[SENSOR FAULT] Node 1: MPU6050 read error. Skipping tilt/vibration parameter.");
        cachedTilt = -999.0f;
        cachedVib  = -999.0f;
    } else {
        cachedTilt = t_angle;
        float total_g = sqrt(ax * ax + ay * ay + az * az);
        cachedVib = fabs(total_g - 1.0f);
    }

    // 3. MANDATORY AUDIT FIX: Ultrasonic background sampling is active during Slot 0 (0-80ms),
    // and ONLY paused when actively inside the Slot 1 transmission window (80-160ms)!
    //
    // FIX (uint32_t underflow): Re-sample millis() HERE, after mpu.update() completes, so that
    // syncLocalTimeMs (which may have been updated by the ESP-NOW ISR during mpu.update()) is
    // always <= currentMs. The ternary guard prevents the rare case where the ISR fires between
    // the millis() call and the subtraction, yielding 0 instead of 4294967295.
    uint32_t currentMs = millis();
    // ATOMIC SNAPSHOT: syncLocalTimeMs and currentSlotDurationMs are written by the
    // ESP-NOW ISR (OnDataRecv). Reading them without disabling interrupts risks a torn
    // read where the ISR updates one but not the other mid-computation.
    noInterrupts();
    uint32_t snap_syncMs  = syncLocalTimeMs;
    uint16_t snap_slotDur = currentSlotDurationMs;
    interrupts();
    uint32_t elapsed = (currentMs >= snap_syncMs) ? (currentMs - snap_syncMs) : 0;
    bool isInsideSlot1Window = (beaconReceived && !slotTransmittedForFrame &&
                                elapsed >= snap_slotDur &&
                                elapsed < (2 * snap_slotDur));

    // INTERRUPT-DRIVEN ULTRASONIC SAMPLING:
    // Two-phase approach: trigger fires the pulse, then after ~40ms we read the ISR-captured result.
    // This completely avoids pulseIn() which is fatally preempted by ESP-NOW Wi-Fi radio ISRs.
    static bool ultrasonicTriggered = false;
    static uint32_t ultrasonicTriggerMs = 0;

    if (!isInsideSlot1Window) {
        if (!ultrasonicTriggered && (currentMs - lastUltrasonicSampleMs >= ULTRASONIC_SAMPLE_INTERVAL_MS)) {
            // Phase 1: Fire the trigger pulse
            triggerUltrasonic();
            ultrasonicTriggered  = true;
            ultrasonicTriggerMs  = currentMs;
        } else if (ultrasonicTriggered && (currentMs - ultrasonicTriggerMs >= 40)) {
            // Phase 2: Re-evaluate slot window with the CURRENT time before reading back,
            // because the slot may have started during the 40ms wait. If we are now inside
            // the TX window, defer the read to the next loop iteration to avoid delaying TX.
            noInterrupts();
            uint32_t snap2_syncMs  = syncLocalTimeMs;
            uint16_t snap2_slotDur = currentSlotDurationMs;
            interrupts();
            uint32_t elapsed2 = (currentMs >= snap2_syncMs) ? (currentMs - snap2_syncMs) : 0;
            bool nowInSlot = (beaconReceived && !slotTransmittedForFrame &&
                              elapsed2 >= snap2_slotDur && elapsed2 < (2 * snap2_slotDur));
            if (!nowInSlot) {
                cachedDisp             = readCachedUltrasonicMm();
                ultrasonicTriggered    = false;
                lastUltrasonicSampleMs = currentMs;
            }
        }
    }

    // 4. INSTANTANEOUS TDMA SLOT 1 TRANSMISSION (< 1 ms dispatch)
    if (beaconReceived && !slotTransmittedForFrame) {
        uint32_t slotStartTime = (uint32_t)NODE_ID * currentSlotDurationMs; // 80ms
        uint32_t slotEndTime   = slotStartTime + currentSlotDurationMs;     // 160ms

        // FIX (uint32_t underflow): Re-sample currentMs again immediately before the slot
        // boundary check so any ESP-NOW ISR that fires between section 3 and here is captured.
        currentMs = millis();
        noInterrupts();
        uint32_t tx_syncMs  = syncLocalTimeMs;
        uint16_t tx_slotDur = currentSlotDurationMs;
        interrupts();
        elapsed = (currentMs >= tx_syncMs) ? (currentMs - tx_syncMs) : 0;
        // Recompute slot boundaries with the freshly snapshotted slot duration
        slotStartTime = (uint32_t)NODE_ID * tx_slotDur;
        slotEndTime   = slotStartTime + tx_slotDur;

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
            if (beaconReceived) {
                // Ensure we don't spam if beacon is lost, though outer if handles this
                Serial.printf("[TDMA GUARD Node %d] Missed window (+%ums, window %u-%ums). Skipping frame.\n",
                              NODE_ID, (unsigned)elapsed, (unsigned)slotStartTime, (unsigned)slotEndTime);
            }
        }
    }
}
