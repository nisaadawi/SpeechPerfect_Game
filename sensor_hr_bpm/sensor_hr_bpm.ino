// ===============================
// XD-58C Fast BPM Monitor
// ===============================

#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <addons/TokenHelper.h>   // Helper for token generation
#include <addons/RTDBHelper.h>    // Helper for RTDB debugging

// Note: tokenStatusCallback is already defined in TokenHelper.h

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

int sensorValue = 0;
int baseline = 2000;
int minSig = 4095;
int maxSig = 0;

// Beat detection
bool lookingForBeat = false;
unsigned long lastBeat = 0;

// Intervals storage (10 beats buffer)
unsigned long intervals[10];
int idx = 0;
int count = 0;

int currentBPM = 0;
float smoothBPM = 0;
int sampleCount = 0;

// ===============================
void setup() {
  Serial.begin(115200);
  pinMode(PULSE_PIN, INPUT);
  delay(500);
  
  Serial.println("\n=== XD-58C Fast BPM Monitor ===");
  
  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\n✅ Connected to WiFi");

  // Initialize Firebase
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback;

  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  
  // Wait for authentication to complete
  Serial.print("Authenticating with Firebase");
  while (auth.token.uid.length() == 0) {
    Serial.print(".");
    delay(100);
  }
  Serial.println("\n✅ Firebase authenticated");
  Serial.print("User UID: ");
  Serial.println(auth.token.uid.c_str());
  
  Serial.println("\nCalibrating for 2 seconds...");
  
  // Faster calibration
  long sum = 0;
  for(int i = 0; i < 200; i++) {
    int r = analogRead(PULSE_PIN);
    sum += r;
    if(r < minSig) minSig = r;
    if(r > maxSig) maxSig = r;
    delay(10);
  }
  baseline = sum / 200;
  
  Serial.print("✓ Baseline: ");
  Serial.println(baseline);
  Serial.print("✓ Range: ");
  Serial.print(minSig);
  Serial.print(" - ");
  Serial.println(maxSig);
  
  Serial.println("\n--- Live Signal (updates every 0.5s) ---");
  delay(500);
}

// ===============================
void loop() {
  sensorValue = analogRead(PULSE_PIN);
  
  // Update baseline slowly (like original)
  baseline = (baseline * 98 + sensorValue * 2) / 100;
  
  // Track min/max for signal graph
  if(sensorValue < minSig) minSig = sensorValue;
  if(sensorValue > maxSig) maxSig = sensorValue;
  
  // Print signal graph every 50 samples (0.5 sec)
  sampleCount++;
  if(sampleCount >= 50) {
    sampleCount = 0;
    
    // Visual bar graph
    Serial.print("Sig:");
    int bars = map(sensorValue, 0, 4095, 0, 50);
    for(int i = 0; i < bars; i++) Serial.print("█");
    Serial.print(" ");
    Serial.print(sensorValue);
    Serial.print(" | Base:");
    Serial.print(baseline);
    Serial.print(" | Diff:");
    Serial.print(sensorValue - baseline);
    Serial.print(" | BPM:");
    Serial.println(currentBPM > 0 ? currentBPM : 0);
    
    // Reset range
    minSig = sensorValue;
    maxSig = sensorValue;
  }
  
  // BEAT DETECTION - XD-58C crosses BELOW baseline
  
  if(sensorValue > baseline + 20) {
    lookingForBeat = true;
  }
  
  if(lookingForBeat && sensorValue < baseline - 20) {
    unsigned long now = millis();
    unsigned long ibi = now - lastBeat;
    
    // Valid beat range (40-150 BPM = 400-1500ms)
    if(ibi > 400 && ibi < 1500 && lastBeat > 0) {
      
      // ORIGINAL'S OUTLIER REJECTION (more robust)
      bool validInterval = true;
      if(count >= 3) {
        unsigned long sumRecent = 0;
        int recentCount = min(count, 5);
        for(int i = 0; i < recentCount; i++) {
          int recentIdx = (idx - 1 - i + 10) % 10;
          sumRecent += intervals[recentIdx];
        }
        float avgRecent = sumRecent / (float)recentCount;
        
        // Reject if more than 40% different from recent average
        if(abs((float)ibi - avgRecent) > avgRecent * 0.4) {
          validInterval = false;
          Serial.print("⚠️  Outlier rejected: ");
          Serial.print(ibi);
          Serial.println("ms");
        }
      }
      
      if(validInterval) {
        // Store interval
        intervals[idx] = ibi;
        idx = (idx + 1) % 10;
        if(count < 10) count++;
        
        // Calculate BPM after 5 beats
        if(count >= 5) {
          unsigned long sumIBI = 0;
          for(int i = 0; i < count; i++) {
            sumIBI += intervals[i];
          }
          float avgIBI = sumIBI / (float)count;
          float rawBPM = 60000.0 / avgIBI;
          
          // Apply smoothing (80/20 like original HRV smoothing)
          if(smoothBPM == 0) {
            smoothBPM = rawBPM;
          } else {
            smoothBPM = smoothBPM * 0.8 + rawBPM * 0.2;
          }
          
          currentBPM = round(smoothBPM);
          
          // Display BPM
          Serial.print("♥ BPM: ");
          Serial.print(currentBPM);
          Serial.print(" | Raw: ");
          Serial.print((int)rawBPM);
          Serial.print(" | IBI: ");
          Serial.print((int)avgIBI);
          Serial.print("ms | Samples: ");
          Serial.println(count);

          // Send BPM to Firebase at hr_read/bpm
          if (Firebase.ready() && (WiFi.status() == WL_CONNECTED)) {
            if (Firebase.RTDB.setInt(&fbdo, "/hr_read/bpm", currentBPM)) {
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
      }
    }
    
    lastBeat = now;
    lookingForBeat = false;
  }
  
  delay(10);
}