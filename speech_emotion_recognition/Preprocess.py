import os 
import numpy as np 
import librosa SR = 16000 
N_MFCC = 40 
FIXED_DURATION = 3 
DATASET_PATH = "dataset/" 
X, y = [], [] label_map = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3} for emotion, label in label_map.items():
 folder = os.path.join(DATASET_PATH, emotion) for file in os.listdir(folder): 
  if file.endswith(".wav"): 
file_path = os.path.join(folder, 
file) signal, sr = 
librosa.load(file_path, sr=SR) 
signal, _ = 
librosa.effects.trim(signal) signal 
= librosa.util.normalize(signal) 
fixed_length = SR * FIXED_DURATION 
signal = librosa.util.fix_length(signal, 
size=fixed_length) mfcc = 
librosa.feature.mfcc(y=signal, sr=sr, 
n_mfcc=N_MFCC) 
mfcc = np.mean(mfcc.T, axis=0) 
X.append(mfcc) 
y.append(label) X = 
np.array(X) y = np.array(y) 
np.save("X_features.npy", 
X) np.save("y_labels.npy", 
y) print("n Preprocessing 
complete!") print("Features 
shape:", X.shape) 
print("Labels shape:", y.shape)