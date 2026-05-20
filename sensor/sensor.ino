#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"
#include <math.h>

// =======================
// Wi-Fi & Server Settings
// =======================
const char* WIFI_SSID     = "pangit_kumunek";
const char* WIFI_PASSWORD = "ivanmapogi";
const char* SERVER_IP     = "192.168.1.224";   // PC/Laptop running Flask
const int   SERVER_PORT   = 5000;             // Flask default port

// =======================
// Sensor Pin Definitions
// =======================
#define DHTPIN    17
#define DHTTYPE   DHT22
#define MQ135_PIN 34
#define MIC_PIN   32

DHT dht(DHTPIN, DHTTYPE);

// =======================
// Variables
// =======================
float mic_rms_counts = 0;
float mic_peak_counts = 0;

// =======================
// Function: Read Microphone RMS and Peak
// =======================
void readMicRMS() {
  const int samples = 500;
  long sumSq = 0;
  int minV = 4095, maxV = 0;
  for (int i = 0; i < samples; i++) {
    int v = analogRead(MIC_PIN);
    sumSq += (long)v * v;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  mic_rms_counts  = sqrt(sumSq / (float)samples);
  mic_peak_counts = maxV - minV;
}

// =======================
// Function: Send JSON data to Flask API
// =======================
void sendEnv(float t, float h, int g) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/api/sensor";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    // Calculate audio dB
    float mic_voltage = mic_rms_counts / 4095.0 * 3.3; // convert ADC counts to voltage
    float audio_db = 20.0 * log10(mic_voltage / 0.006 + 1e-6); // avoid log(0)

    // Build JSON payload
    String payload = "{";
    payload += "\"user_id\":1,";
    payload += "\"device_id\":\"esp32_001\",";
    payload += "\"temperature\":" + String(t, 1) + ",";
    payload += "\"humidity\":" + String(h, 1) + ",";
    payload += "\"gas\":" + String(g) + ",";
    payload += "\"mic_rms\":" + String(mic_rms_counts, 1) + ",";
    payload += "\"mic_peak\":" + String(mic_peak_counts, 1) + ",";
    payload += "\"audio_level_db\":" + String(audio_db, 1);
    payload += "}";

    // Send HTTP POST
    int code = http.POST(payload);
    if (code > 0) {
      Serial.printf("[ENV] POST code: %d | Response: %s\n", code, http.getString().c_str());
    } else {
      Serial.printf("[ENV] POST failed, error: %s\n", http.errorToString(code).c_str());
    }
    http.end();
  } else {
    Serial.println("[WiFi] Disconnected, skipping send.");
  }
}

// =======================
// Setup
// =======================
void setup() {
  Serial.begin(115200);
  dht.begin();

  analogSetPinAttenuation(MIC_PIN, ADC_11db);
  analogSetPinAttenuation(MQ135_PIN, ADC_11db);

  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 30) {
    delay(500);
    Serial.print(".");
    retry++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected");
    Serial.print("📡 IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect WiFi");
  }
}

// =======================
// Main Loop
// =======================
void loop() {
  // Reconnect Wi-Fi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Reconnecting...");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    delay(1000);
    return;
  }

  // Read sensors
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  int g = analogRead(MQ135_PIN);
  readMicRMS();

  if (isnan(t) || isnan(h)) {
    Serial.println("[DHT] Reading failed, skipping...");
  } else {
    // Debug print
    Serial.printf("🌡️ Temp: %.1f°C | 💧 Hum: %.1f%% | 🧪 Gas: %d | 🔊 RMS: %.1f | Peak: %.1f\n",
                  t, h, g, mic_rms_counts, mic_peak_counts);

    // Send to Flask
    sendEnv(t, h, g);
  }

 // Send data every 1 minute (60 000 ms)
delay(60 * 1000);

}
