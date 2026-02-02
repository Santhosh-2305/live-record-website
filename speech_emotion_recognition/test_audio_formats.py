import librosa
import numpy as np
import os

def test_audio_format(file_path):
    """Test if an audio file can be processed"""
    try:
        print(f"\n=== Testing: {file_path} ===")
        
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
        
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Try loading with librosa
        audio, sr = librosa.load(file_path, sr=22050)
        print(f"Loaded successfully: {len(audio)} samples at {sr} Hz")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0)
        print(f"MFCC features extracted: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("Testing audio format support...")
    
    # Test the generated WAV file
    test_files = [
        "test_audio.wav"
    ]
    
    for file_path in test_files:
        success = test_audio_format(file_path)
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    print("\n=== Audio Format Support Test Complete ===")
    print("If WAV works, the system should handle most formats.")
    print("For MP3/MP4 files, make sure you have ffmpeg installed.")

if __name__ == "__main__":
    main()