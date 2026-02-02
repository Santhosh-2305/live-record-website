import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] 
= "0" 
os.environ["TF_USE_LEGACY_KERAS"] = 
"1" import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, 
Dropout from tensorflow.keras.utils import to_categorical 
X = 
52
np.load("X_features.npy
") y = 
np.load("y_labels.npy") 
print("Data loaded:", X.shape, y.shape) 
X = X.astype("float32") 
X = X.reshape(X.shape[0], X.shape[1], 1) y = 
to_categorical(y, 
num_classes=4).astype("float32") 
model = Sequential([ 
Conv1D(64, kernel_size=3, activation='relu', input_shape=(40, 1)), 
MaxPooling1D(pool_size=2), 
LSTM(64, return_sequences=False), 
Dropout(0.3), 
Dense(64, activation='relu'), 
Dense(4, activation='softmax') 
]) 
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy']) history = model.fit(X, y, epochs=30, batch_size=32, 
validation_split=0.2) model.save("emotion_model.h5") 
print("n Model training complete! Saved as emotion_model.h5")