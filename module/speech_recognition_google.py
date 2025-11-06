import speech_recognition as sr
import numpy as np
import audioop
import time
from IPython.display import clear_output

# 🎙️ Initialize recognizer and microphone
recognizer = sr.Recognizer()
mic = sr.Microphone()

print("Adjusting for ambient noise... Please wait.")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
print("✅ Adjustment complete. Start speaking!\n")

# ===============================
# Initialize Metrics
# ===============================
full_transcript = ""
start_time = time.time()
word_count = 0
pause_times = []
last_speech_end = time.time()
filler_count = 0

# RMS and Pitch Calculation
def calculate_rms(audio_data):
    return audioop.rms(audio_data.get_raw_data(), 2)

def estimate_pitch(audio_data):
    data = np.frombuffer(audio_data.get_raw_data(), np.int16)
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    duration = len(data) / audio_data.sample_rate
    if duration > 0:
        return len(zero_crossings) / (2.0 * duration)
    return 0

# Detect Fillers
FILLER_WORDS = {"um", "uh", "erm", "hmm", "like", "you know", "so", "well", "basically", "actually"}

def count_fillers(text):
    words = text.lower().split()
    return sum(1 for w in words if w in FILLER_WORDS)

try:
    while True:
        with mic as source:
            print("🎤 Listening...")
            audio = recognizer.listen(source, phrase_time_limit=5)

        try:
            text = recognizer.recognize_google(audio)
            full_transcript += text + "\n"

            # Update Metrics
            rms = calculate_rms(audio)
            pitch = estimate_pitch(audio)
            words = len(text.split())
            word_count += words

            fillers = count_fillers(text)
            filler_count += fillers

            now = time.time()
            pause_duration = now - last_speech_end
            if pause_duration > 1.0:
                pause_times.append(pause_duration)
            last_speech_end = now

            # Refresh Live Output
            clear_output(wait=True)
            print("📝 Live Transcript:\n")
            print(full_transcript)
            print("-" * 40)
            print(f"📊 Metrics (Latest Segment):")
            print(f"   • RMS (Loudness): {rms:.2f}")
            print(f"   • Pitch: {pitch:.1f} Hz")
            print(f"   • Words this round: {words}")
            print(f"   • Fillers this round: {fillers}")
            if pause_times:
                print(f"   • Last pause: {pause_duration:.2f}s")
            print("-" * 40)
            print(f"🧮 Cumulative Stats:")
            print(f"   • Total words: {word_count}")
            print(f"   • Total fillers: {filler_count}")
            print("=" * 40)

        except sr.UnknownValueError:
            print("❌ Could not understand audio. Try again.\n")
            time.sleep(1)

        except sr.RequestError as e:
            print("⚠️ API error:", e)
            break

except KeyboardInterrupt:
    # When stopped manually
    total_time = time.time() - start_time
    wpm = (word_count / total_time) * 60 if total_time > 0 else 0
    avg_pause = np.mean(pause_times) if pause_times else 0
    filler_rate = (filler_count / word_count) * 100 if word_count > 0 else 0

    print("\n🛑 Recording stopped by user.")
    print("=" * 40)
    print(f"🧮 Final Performance Summary:")
    print(f"   • Total Words: {word_count}")
    print(f"   • Speaking Duration: {total_time:.1f}s")
    print(f"   • WPM (Words per Minute): {wpm:.1f}")
    print(f"   • Avg Pause Duration: {avg_pause:.2f}s")
    print(f"   • Total Fillers: {filler_count}")
    print(f"   • Filler Rate: {filler_rate:.1f}% of words")
    print("=" * 40)
