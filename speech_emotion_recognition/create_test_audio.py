import numpy as np
import librosa
import soundfile as sf
import os

def create_test_audio():
    """Create a simple test audio file for demonstration"""
    
    # Create a simple sine wave (like a tone)
    duration = 3  # seconds
    sample_rate = 22050
    frequency = 440  # A4 note
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple audio signal (sine wave with some variation)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some variation to make it more speech-like
    audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.1
    audio += np.random.normal(0, 0.05, len(audio))  # Add some noise
    
    # Apply envelope to make it more natural
    envelope = np.exp(-t * 0.5)  # Decay envelope
    audio = audio * envelope
    
    # Save as WAV file
    output_file = "test_audio.wav"
    sf.write(output_file, audio, sample_rate)
    
    print(f"Test audio file created: {output_file}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Test feature extraction
    features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    features_mean = np.mean(features.T, axis=0)
    
    print(f"MFCC features shape: {features_mean.shape}")
    print("You can now upload this file to test your web app!")

if __name__ == "__main__":
    create_test_audio()