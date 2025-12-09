const int PULSE_PIN = 36;
unsigned long lastBeat = 0;

int bpmHistory[8];
int bpmIndex = 0;
int stableBPM = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("Wrist Pulse BPM (Smoothed)");
}

void loop() {
  int raw = analogRead(PULSE_PIN);
  static int prev = raw;
  static bool rising = false;
  static int threshold = 2000;

  threshold = (threshold * 9 + raw) / 10;

  if (raw > threshold + 20 && !rising) rising = true;

  if (raw < threshold && rising) {
    unsigned long now = millis();
    unsigned long interval = now - lastBeat;

    if (interval > 350 && interval < 2000) {
      int bpm = 60000 / interval;

      // ✅ Restrict realistic range while speaking
      if (bpm >= 80 && bpm <= 150) {
        bpmHistory[bpmIndex] = bpm;
        bpmIndex = (bpmIndex + 1) % 8;

        int sum = 0;
        for (int i = 0; i < 8; i++) sum += bpmHistory[i];
        stableBPM = sum / 8;

        Serial.print("🖐️ Wrist BPM: ");
        Serial.println(stableBPM);
      }
    }
    lastBeat = now;
    rising = false;
  }

  prev = raw;
  delay(10);
}
