/**
 * ====================================================================================================
 * PROJECT: INTRINSICALLY SAFE REAL-TIME MINE SUBSIDENCE MONITORING (SIH 26025)
 * FIRMWARE: Sensor Field Node 2 (sensor_node_esp32wroom_4.ino)
 * TARGET: ESP-WROOM-32 Development Board (Arduino IDE Compilable)
 * ====================================================================================================
 * 
 * AUDIT CORRECTIONS IMPLEMENTED:
 * 1. Hardware Pin Reassignment (CRITICAL):
 *    - Reassigned PIN_DIGITAL_VIBRATION from GPIO 5 to GPIO 16.
 *    - GPIO 5 is an ESP32 hardware strapping pin (MTDI); connecting an external sensor to GPIO 5
 *      risks bootloader/UART strapping failures upon power-up or reset. GPIO 16 is completely safe.
 * 2. Hardware ISR for Instantaneous Vibration Pulse Capture (CRITICAL):
 *    - Replaced polling in loop() with a dedicated hardware interrupt (FALLING edge).
 *    - Captures transient sub-millisecond mechanical shock pulses (10–500 µs) instantly, even during
 *      transmit microsecond delays.
 *    - In transmitTelemetry(), mapped payload.vibration = vibrationLatched ? 1.0f : 0.0f; and reset
 *      vibrationLatched = false immediately post-transmission.
 * 3. Explicit RF Wi-Fi Channel Lock:
 *    - Included <esp_wifi.h>.
 *    - esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE); locks RF hardware to Ch 1.
 * 4. Pure Digital Tilt Switch:
 *    - Sampled on GPIO 4 with internal pull-up (INPUT_PULLUP).
 *    - On trigger, mapped to payload.tilt = 10.0f (instantly breaches Central Hub 5.0f threshold).
 * 5. Strict Slot Window Upper-Bound Guard:
 *    - Node 2 transmits strictly within Slot 2 (160ms to 240ms).
 * ====================================================================================================
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h> // Explicit Wi-Fi Hardware Channel Configuration

/* ====================================================================================================
 * 1. NODE CONFIGURATION & HARDWARE PIN ASSIGNMENTS
 * ==================================================================================================== */
#define NODE_ID                 2    // NODE 2 (Assigned Slot 2: 160ms to 240ms)
#define ESPNOW_WIFI_CHANNEL     1    // Must strictly match Central Hub channel (Locked)

// Digital Sensor Pins (ESP-WROOM-32)
#define PIN_DIGITAL_TILT        4    // Digital Tilt Sensor DO Pin (e.g., SW-520D ball switch)
#define PIN_DIGITAL_VIBRATION   16   // Reassigned from GPIO 5 to GPIO 16 (Avoids ESP32 strapping issue)
#define PIN_STATUS_LED          2    // Onboard Status LED

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

// MANDATORY AUDIT FIX: Hardware ISR Latched Flag
// Captures sub-millisecond mechanical vibration pulses via hardware interrupt
volatile bool vibrationLatched = false;

/* ====================================================================================================
 * 4. HARDWARE INTERRUPT SERVICE ROUTINE FOR SW-420 VIBRATION SENSOR
 * ==================================================================================================== */
void IRAM_ATTR vibrationISR() {
    vibrationLatched = true;
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
 * 6. TELEMETRY TRANSMISSION (EXECUTED STRICTLY WITHIN ASSIGNED TDMA SLOT 2)
 * ==================================================================================================== */
void transmitTelemetry() {
    // Read Digital Tilt state
    bool tiltTriggered = (digitalRead(PIN_DIGITAL_TILT) == SENSOR_TRIGGER_LEVEL);

    SensorPayload payload;
    payload.msg_type = PKT_TYPE_TELEMETRY;
    payload.node_id = NODE_ID;

    // Map Digital Tilt: If triggered, send 10.0f to instantly exceed Central Hub's 5.0f limit
    payload.tilt = tiltTriggered ? 10.0f : 0.0f;

    // Map Hardware ISR Vibration Latch: Atomically capture and clear latch for this superframe
    noInterrupts();
    bool currentVib = vibrationLatched;
    vibrationLatched = false;
    interrupts();

    payload.vibration = currentVib ? 1.0f : 0.0f;

    // Displacement fixed to 0.0f on Node 2
    payload.displacement = 0.0f;

    esp_err_t result = esp_now_send(hubBroadcastAddress, (uint8_t *)&payload, sizeof(payload));
    if (result == ESP_OK) {
        Serial.printf("[NODE %d TX SLOT 2] Digital Telemetry: Tilt=%04.1f deg (%s) | Vib=%04.1f (%s) | Disp=0.0mm\n",
                      NODE_ID,
                      payload.tilt, (payload.tilt > 0.0f ? "TRIGGERED" : "IDLE"),
                      payload.vibration, (payload.vibration > 0.0f ? "LATCHED VIB" : "CALM"));

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
    Serial.printf("\n[NODE %d] Initializing Digital Sensor Node (sensor_node_esp32wroom_4)...\n", NODE_ID);

    // Initialize Status LED
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    // Initialize Digital Tilt and Vibration Sensors with Internal Pull-Ups
    pinMode(PIN_DIGITAL_TILT, INPUT_PULLUP);
    pinMode(PIN_DIGITAL_VIBRATION, INPUT_PULLUP);
    Serial.println("[SENSOR] Digital Tilt (GPIO 4) & Vibration (GPIO 16) configured with INPUT_PULLUP.");

    // MANDATORY AUDIT FIX: Attach hardware interrupt on GPIO 16 (FALLING edge)
    // Guarantees sub-millisecond mechanical shock pulses are never missed by loop delays
    attachInterrupt(digitalPinToInterrupt(PIN_DIGITAL_VIBRATION), vibrationISR, FALLING);
    Serial.println("[ISR] Attached vibrationISR on GPIO 16 (FALLING edge).");

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
 * 8. MAIN LOOP: STRICT TDMA DISPATCH WITH HARDWARE ISR CAPTURE
 * ==================================================================================================== */
void loop() {
    // Note: SW-420 pulses are now captured by hardware interrupt vibrationISR() instantly.
    // No polling delay or blocking can cause vibration pulses to be missed!

    // TDMA Slot 2 Transmission Check
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
            Serial.printf("[TDMA GUARD Node %d] Missed window (+%lums, window %lu-%lums). Skipping frame.\n",
                          NODE_ID, elapsedSinceBeacon, slotStartTime, slotEndTime);
        }
    }
}
