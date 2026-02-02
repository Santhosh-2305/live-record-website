import os 
import numpy as np 
import librosa 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout 
from tensorflow.keras.utils import to_categorical 

# Configuration
SR = 16000 
N_MELS = 40 
EMOTIONS = ["neutral", "happy", "sad", "angry"]

def extract_features(file_path): 
    """Extract MFCC features from audio file"""
    audio, sr = librosa.load(file_path, sr=SR) 
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MELS) 
    return np.mean(mfccs.T, axis=0) 

def load_data():
    """Load and preprocess data from dataset"""
    X, y = [], [] 
    dataset_path = "dataset" 
    
    for idx, emotion in enumerate(EMOTIONS): 
        emotion_path = os.path.join(dataset_path, emotion) 
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion_path} does not exist")
            continue
            
        for file in os.listdir(emotion_path): 
            if file.endswith(".wav"): 
                file_path = os.path.join(emotion_path, file) 
                try:
                    features = extract_features(file_path) 
                    X.append(features) 
                    y.append(idx)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    """Create CNN-LSTM model"""
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, emotions):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=emotions, 
                yticklabels=emotions, 
                cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    """Main training pipeline"""
    print("Loading data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found! Please check your dataset folder.")
        return
    
    print(f"Data loaded: {X.shape[0]} samples")
    
    # Reshape for CNN input
    X = X[..., np.newaxis]
    y_cat = to_categorical(y, num_classes=len(EMOTIONS))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create and train model
    model = create_model((X.shape[1], 1), len(EMOTIONS))
    
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=30, 
        batch_size=16,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model.save("speech_emotion_model.h5")
    print("Model saved as speech_emotion_model.h5")
    
    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, EMOTIONS)
    
    print("Training complete!")

if __name__ == "__main__":
    main()