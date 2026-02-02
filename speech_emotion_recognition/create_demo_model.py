import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# Create some dummy data for demonstration
print("Creating demo model with synthetic data...")

# Generate synthetic MFCC features (40 features per sample)
np.random.seed(42)
n_samples = 1000
n_features = 40
n_classes = 4

# Create synthetic features
X = np.random.randn(n_samples, n_features)

# Create synthetic labels (0=Angry, 1=Happy, 2=Sad, 3=Neutral)
y = np.random.randint(0, n_classes, n_samples)

# Convert to categorical
y_categorical = to_categorical(y, num_classes=n_classes)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y_categorical.shape}")

# Create a simple model
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training demo model...")
history = model.fit(
    X, y_categorical,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the model
model.save("model/emotion_model.h5")
print("Demo model saved successfully!")

# Test the model
test_sample = np.random.randn(1, n_features)
prediction = model.predict(test_sample)
emotions = ["Angry", "Happy", "Sad", "Neutral"]
predicted_emotion = emotions[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Test prediction: {predicted_emotion} ({confidence:.1f}% confidence)")