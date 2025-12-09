// ===============================
// XD-58C Signal Visualizer + HRV (FAST START)
// ===============================

const int PULSE_PIN = 36;

int sensorValue = 0;
int prevValue = 0;
int baseline = 2000;
int minSig = 4095;
int maxSig = 0;

// Beat detection
bool lookingForBeat = false;
unsigned long lastBeat = 0;

// Intervals storage
unsigned long intervals[15];
int idx = 0;
int count = 0;

// HRV smoothing
float smoothRMSSD = 0;
float smoothSDNN = 0;

int sampleCount = 0;

// ===============================
void setup() {
  Serial.begin(115200);
  pinMode(PULSE_PIN, INPUT);
  delay(1000);
  
  Serial.println("\n=== XD-58C HRV Monitor (FAST) ===");
  Serial.println("Calibrating for 2 seconds...");
  
  // FASTER Calibrate (reduced from 3s to 2s)
  long sum = 0;
  for(int i = 0; i < 200; i++) {  // Changed from 300 to 200
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
  Serial.print("✓ Swing: ");
  Serial.println(maxSig - minSig);
  
  if(maxSig - minSig < 50) {
    Serial.println("\n⚠️  Signal too weak - check finger placement!");
  }
  
  Serial.println("\n--- Live Signal (updates every 0.5s) ---");
  delay(500);  // Reduced from 1000ms
}

// ===============================
void loop() {
  prevValue = sensorValue;
  sensorValue = analogRead(PULSE_PIN);
  
  // Update baseline slowly
  baseline = (baseline * 98 + sensorValue * 2) / 100;
  
  // Track min/max
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
    Serial.print(" | Beats:");
    Serial.println(count);
    
    // Reset range
    minSig = sensorValue;
    maxSig = sensorValue;
  }
  
  // BEAT DETECTION
  // XD-58C: look for signal crossing BELOW baseline
  
  if(sensorValue > baseline + 20) {
    lookingForBeat = true;
  }
  
  if(lookingForBeat && sensorValue < baseline - 20) {
    unsigned long now = millis();
    unsigned long ibi = now - lastBeat;
    
    if(ibi > 400 && ibi < 1500 && lastBeat > 0) {
      
      // RELAXED outlier rejection - only after 3 beats
      bool validInterval = true;
      if(count >= 3) {
        unsigned long sumRecent = 0;
        int recentCount = min(count, 5);
        for(int i = 0; i < recentCount; i++) {
          int recentIdx = (idx - 1 - i + 15) % 15;
          sumRecent += intervals[recentIdx];
        }
        float avgRecent = sumRecent / (float)recentCount;
        
        // More lenient: 50% instead of 40%
        if(abs((float)ibi - avgRecent) > avgRecent * 0.5) {
          validInterval = false;
          Serial.print("⚠️  Outlier rejected: ");
          Serial.print(ibi);
          Serial.println("ms");
        }
      }
      
      if(validInterval) {
        // Valid beat!
        intervals[idx] = ibi;
        idx = (idx + 1) % 15;
        if(count < 15) count++;
        
        // KEY CHANGE: Show BPM after just 3 beats instead of 8!
        if(count >= 3) {
          // Calculate BPM from ALL intervals
          unsigned long sumIBI = 0;
          for(int i = 0; i < count; i++) sumIBI += intervals[i];
          int bpm = 60000 / (sumIBI / count);
          
          // Calculate HRV metrics (but show "calculating..." if < 5 beats)
          if(count >= 5) {
            // Calculate SDNN
            float meanIBI = sumIBI / (float)count;
            float varSum = 0;
            for(int i = 0; i < count; i++) {
              float d = intervals[i] - meanIBI;
              varSum += d * d;
            }
            float sdnn = sqrt(varSum / count);
            
            // Calculate RMSSD
            float diffSum = 0;
            for(int i = 1; i < count; i++) {
              float d = (float)intervals[i] - (float)intervals[i-1];
              diffSum += d * d;
            }
            float rmssd = sqrt(diffSum / (count - 1));
            
            // Smooth
            if(smoothRMSSD == 0) {
              smoothRMSSD = rmssd;
              smoothSDNN = sdnn;
            } else {
              smoothRMSSD = smoothRMSSD * 0.8 + rmssd * 0.2;
              smoothSDNN = smoothSDNN * 0.8 + sdnn * 0.2;
            }
            
            Serial.print(">>> BPM:");
            Serial.print(bpm);
            Serial.print(" | SDNN:");
            Serial.print(smoothSDNN, 1);
            Serial.print("ms | RMSSD:");
            Serial.print(smoothRMSSD, 1);
            Serial.print("ms <<<");
            Serial.println();
          } else {
            // Show BPM early, but indicate HRV is still calculating
            Serial.print(">>> BPM:");
            Serial.print(bpm);
            Serial.print(" | HRV: calculating... (");
            Serial.print(count);
            Serial.print("/5 beats) <<<");
            Serial.println();
          }
        }
      }
    }
    
    lastBeat = now;
    lookingForBeat = false;
  }
  
  delay(10);
}