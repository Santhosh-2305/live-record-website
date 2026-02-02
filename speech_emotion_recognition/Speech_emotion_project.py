import os 
import 
numpy as np 
import 
librosa 
from sklearn.model_selection import train_test_split from 
sklearn.metrics import confusion_matrix import seaborn as 
sns import matplotlib.pyplot as plt from 
tensorflow.keras.models import Sequential from 
tensorflow.keras.layers import Conv1D, LSTM, Dense, 
Dropout from tensorflow.keras.utils import to_categorical 
SR = 16000 
N_MELS = 40 
EMOTIONS = ["neutral", "happy", "sad", 
"angrys"] def extract_features(file_path): 
audio, sr = librosa.load(file_path, sr=SR) 
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MELS) 
return np.mean(mfccs.T, axis=0) X, 
y = [], [] dataset_path = "dataset" 
for idx, emotion in 
enumerate(EMOTIONS): 
emotion_path = os.path.join(dataset_path, 
emotion) for file in 
os.listdir(emotion_path): if 
file.endswith(".wav"): file_path = 
os.path.join(emotion_path, file) features = 
extract_features(file_path) 
X.append(features) 
y.append(idx) 
X = 
np.array(X) y = 
np.array(y) X = 
X[..., 
np.newaxis] 
y_cat = to_categorical(y, num_classes=len(EMOTIONS)) 
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, 
random_state=42) model = Sequential([ 
Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)), 
51
Dropout(0.2), 
LSTM(128), 
Dense(64, activation='relu'), 
Dense(len(EMOTIONS), activation='softmax') 
]) 
model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) history = model.fit(X_train, y_train, validation_data=(X_test, 
y_test), epochs=30, batch_size=16) plt.figure(figsize=(12,5)) plt.subplot(1,2,1) 
plt.plot(history.history['accuracy'], label='train_acc') 
plt.plot(history.history['val_accuracy'], label='val_acc') plt.title('Model Accuracy') 
plt.xlabel('Epochs') plt.ylabel('Accuracy') plt.legend() plt.grid() plt.subplot(1,2,2) 
plt.plot(history.history['loss'], 
label='train_loss') 
plt.plot(history.history['val_loss'], 
label='val_loss') plt.title('Model Loss') 
plt.xlabel('Epochs') plt.ylabel('Loss') 
plt.legend() plt.grid() plt.show() 
model.save("speech_emotion_model.h5") 
print("n Model saved as 
speech_emotion_model.h5") y_pred = 
np.argmax(model.predict(X_test), axis=1) 
y_true = np.argmax(y_test, axis=1) cm = 
confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(6,5)) 
sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, 
yticklabels=EMOTIONS, cmap="Blues") plt.ylabel("True Label") 
plt.xlabel("Predicted Label") plt.title("Confusion Matrix") plt.show() 
print("n Plots and confusion matrix saved as PNG files.")