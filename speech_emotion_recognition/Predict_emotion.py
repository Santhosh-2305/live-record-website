import numpy as np 
import librosa 
from tensorflow.keras.models import 
load_model SR = 16000 
model = 
load_model("emotion_model.h5") print("n 
Model loaded!") 
EMOTIONS = ["neutral", "happy", "sad", 
"angry"] def extract_features(file_path, sr=SR): 
audio, sr = librosa.load(file_path, sr=sr) mfccs 
= librosa.feature.mfcc(y=audio, sr=sr, 
n_mfcc=40) mfccs = np.mean(mfccs.T, axis=0) 
return mfccs def predict_emotion(file_path): 
features = extract_features(file_path) preds = 
model.predict(features[np.newaxis, ...]) 
predicted_label = np.argmax(preds) return 
preds, predicted_label if __name__ == 
"__main__": test_file = "test_audio/gopi.wav" 
preds, label = predict_emotion(test_file) 
print("n Probabilities:", preds) print("n 
53
Predicted Emotion:", EMOTIONS[label]) 
top_indices = preds[0].argsort()[-2:][::-1] 
print("\nTop 2 emotions:") for i in top_indices: 
print(f"{EMOTIONS[i]}: {preds[0][i]:.2f}")