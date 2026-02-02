import sounddevice 
as sd import numpy 
as np import librosa 
from tensorflow.keras.models import load_model 
SR = 16000 
DURATION = 2 
N_MFCC = 40 
MODEL_PATH = "speech_emotion_model.h5" 
EMOTIONS = ['neutral', 'happy', 'sad', 'angry'] 
print("Loading model...") 
model = load_model(MODEL_PATH) 
print("n Model loaded!") def
record_audio(duration=DURATION, 
sr=SR): 
print("Recording...") 
audio = sd.rec(int(duration * sr), samplerate=sr, 
channels=1) sd.wait() audio = audio.flatten() 
print("Recording complete") 
return audio def extract_features(y, 
sr=SR, n_mfcc=N_MFCC): 
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) 
mfccs = np.mean(mfccs.T, axis=0) return mfccs 
print("\nStarting real-time emotion detection. Press Ctrl+C 
to stop.\n") try: while True: 
audio = record_audio() features = 
extract_features(audio) features = 
features.reshape(1, -1) prediction = 
model.predict(features) predicted_emotion = 
EMOTIONS[np.argmax(prediction)] 
50
print(f"Predicted Emotion: 
{predicted_emotion}\n") except 
KeyboardInterrupt: 
print("\nStopped real-time detection.")