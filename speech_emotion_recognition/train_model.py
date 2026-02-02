import os
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

emotions = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3
}

X = []
y = []

def extract(file):
    audio, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

for emotion in emotions:
    path = os.path.join("dataset", emotion)
    for file in os.listdir(path):
        features = extract(os.path.join(path, file))
        X.append(features)
        y.append(emotions[emotion])

X = np.array(X)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(256, activation="relu", input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=30, batch_size=32)

model.save("model/emotion_model.h5")

print("MODEL SAVED SUCCESSFULLY")
