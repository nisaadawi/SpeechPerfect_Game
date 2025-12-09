#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <addons/TokenHelper.h>   // Helper for token generation
#include <addons/RTDBHelper.h>    // Helper for RTDB debugging

// Token status callback function
void tokenStatusCallback(TokenInfo info) {
  if (info.status == token_status_error) {
    Serial.printf("Token info: type = %s, status = %s\n", getTokenType(info).c_str(), getTokenStatus(info).c_str());
    Serial.printf("Token error: %s\n\n", info.error.c_str());
  }
}

#define WIFI_SSID "Wereen z"
#define WIFI_PASSWORD "wereenzz"
#define API_KEY "AIzaSyB92lB8qxZJjFfsB5rOPugnUAgflEM4MlE"
#define DATABASE_URL "https://speechperfect-b00c7-default-rtdb.firebaseio.com/"

// Firebase Authentication credentials
#define USER_EMAIL "wereen0909@gmail.com"
#define USER_PASSWORD "Wereenz0909?"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

const int PULSE_PIN = 36;
int Signal;
int Threshold = 2000; // Adaptive threshold starting point

unsigned long lastBeat = 0;
unsigned long beatTimes[10];
int beatIndex = 0;
bool rising = false;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Accurate Finger BPM starting...");
  
  // Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\n✅ Connected to WiFi");

  // Initialize Firebase
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback; // Add token status callback

  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  
  // Wait for authentication to complete
  Serial.print("Authenticating with Firebase");
  while (auth.token.uid == "") {
    Serial.print(".");
    delay(100);
  }
  Serial.println("\n✅ Firebase authenticated");
  Serial.print("User UID: ");
  Serial.println(auth.token.uid);
}

void loop() {
  Signal = analogRead(PULSE_PIN);

  static int prev = Signal;

  // Adaptive threshold movement
  Threshold = (Threshold * 9 + Signal) / 10;

  if (Signal > Threshold + 25 && !rising) {
    rising = true;
  }

  if (Signal < Threshold && rising) {
    unsigned long now = millis();
    unsigned long interval = now - lastBeat;

    if (interval > 350 && interval < 2000) { // Only valid heart intervals
      beatTimes[beatIndex] = interval;
      beatIndex = (beatIndex + 1) % 10;

      long avgInterval = 0;
      for (int i = 0; i < 10; i++) {
        avgInterval += beatTimes[i];
      }
      avgInterval /= 10;

      int bpm = 60000 / avgInterval;
      Serial.print("💓 BPM: ");
      Serial.println(bpm);

      // Send BPM to Firebase at hr_read/bpm
      if (Firebase.ready() && (WiFi.status() == WL_CONNECTED)) {
        if (Firebase.RTDB.setInt(&fbdo, "/hr_read/bpm", bpm)) {
          Serial.println("✅ Sent BPM to Firebase");
        } else {
          Serial.print("❌ Firebase Error: ");
          Serial.print(fbdo.errorReason());
          Serial.print(" (Code: ");
          Serial.print(fbdo.httpCode());
          Serial.println(")");
          
          // Try to reconnect if authentication failed
          if (fbdo.httpCode() == 401 || fbdo.httpCode() == 403) {
            Serial.println("⚠️ Authentication issue. Attempting to reconnect...");
            Firebase.reconnectWiFi(true);
            delay(1000);
          }
        }
      } else {
        if (!Firebase.ready()) {
          Serial.println("⚠️ Firebase not ready");
        }
        if (WiFi.status() != WL_CONNECTED) {
          Serial.println("⚠️ WiFi disconnected");
        }
      }
    }

    lastBeat = now;
    rising = false;
  }

  prev = Signal;
  delay(10);
}