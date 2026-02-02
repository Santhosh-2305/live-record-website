import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LSTM, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import os

def create_full_emotion_model():
    """Create a model that supports all 7 RAVDESS emotions"""
    
    print("Creating comprehensive emotion model with 7 emotions...")
    
    # RAVDESS emotion mapping
    emotions = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
    n_emotions = len(emotions)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 2000
    n_features = 40
    
    # Create synthetic MFCC features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels with realistic distribution
    # More samples for common emotions (Happy, Sad, Angry, Neutral)
    emotion_weights = [0.2, 0.25, 0.25, 0.2, 0.05, 0.03, 0.02]  # Distribution weights
    y = np.random.choice(n_emotions, n_samples, p=emotion_weights)
    
    # Convert to categorical
    y_categorical = to_categorical(y, num_classes=n_emotions)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y_categorical.shape}")
    print(f"Emotion distribution:")
    for i, emotion in enumerate(emotions):
        count = np.sum(y == i)
        print(f"  {emotion}: {count} samples ({count/n_samples*100:.1f}%)")
    
    # Reshape for CNN input
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Create advanced model architecture
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_features, 1)),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # LSTM for temporal patterns
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_emotions, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X, y_categorical,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model.save("model/emotion_model_7class.h5")
    print(f"\nModel saved as emotion_model_7class.h5")
    
    # Test prediction
    test_sample = np.random.randn(1, n_features, 1)
    prediction = model.predict(test_sample, verbose=0)
    predicted_emotion = emotions[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"\nTest prediction: {predicted_emotion} ({confidence:.1f}% confidence)")
    
    # Show all probabilities
    print("\nAll emotion probabilities:")
    for i, emotion in enumerate(emotions):
        prob = prediction[0][i] * 100
        print(f"  {emotion}: {prob:.1f}%")
    
    return model, emotions

if __name__ == "__main__":
    model, emotions = create_full_emotion_model()