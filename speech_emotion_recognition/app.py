from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
import subprocess
import sys

app = Flask(__name__)

# load trained model
model = load_model("model/emotion_model_7class.h5")

emotions = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg if available"""
    try:
        # Try using ffmpeg for conversion
        result = subprocess.run([
            'ffmpeg', '-i', input_path, '-ar', '22050', '-ac', '1', 
            '-f', 'wav', output_path, '-y'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg conversion failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("FFmpeg not found, trying direct librosa loading")
        return False

def extract_features(file_path):
    """Extract MFCC features from audio file with multiple format support"""
    try:
        print(f"Attempting to load audio file: {file_path}")
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"File extension: {file_ext}")
        
        # Try different approaches based on file type
        audio = None
        sr = None
        
        try:
            # Method 1: Direct librosa loading (works for most formats)
            print("Trying direct librosa loading...")
            audio, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
            print(f"Direct loading successful. Audio length: {len(audio)}, Sample rate: {sr}")
            
        except Exception as e1:
            print(f"Direct loading failed: {e1}")
            
            try:
                # Method 2: Load without duration/offset constraints
                print("Trying librosa loading without constraints...")
                audio, sr = librosa.load(file_path, sr=22050)
                
                # Apply duration and offset manually if needed
                if len(audio) > sr * 3:  # If longer than 3 seconds
                    start_sample = int(0.5 * sr)  # 0.5 second offset
                    end_sample = start_sample + int(3 * sr)  # 3 seconds duration
                    audio = audio[start_sample:end_sample]
                
                print(f"Constrained loading successful. Audio length: {len(audio)}, Sample rate: {sr}")
                
            except Exception as e2:
                print(f"Constrained loading failed: {e2}")
                
                try:
                    # Method 3: Try loading with different parameters
                    print("Trying librosa loading with mono conversion...")
                    audio, sr = librosa.load(file_path, sr=None, mono=True)
                    
                    # Resample to 22050 if needed
                    if sr != 22050:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                        sr = 22050
                    
                    # Apply duration constraints
                    if len(audio) > sr * 3:
                        start_sample = int(0.5 * sr)
                        end_sample = start_sample + int(3 * sr)
                        audio = audio[start_sample:end_sample]
                    
                    print(f"Mono conversion successful. Audio length: {len(audio)}, Sample rate: {sr}")
                    
                except Exception as e3:
                    print(f"Mono conversion failed: {e3}")
                    raise Exception(f"Could not load audio file. Tried multiple methods. Last error: {str(e3)}")
        
        # Ensure we have audio data
        if audio is None or len(audio) == 0:
            raise ValueError("No audio data found in file")
        
        # Ensure minimum length
        if len(audio) < sr * 0.1:  # Less than 0.1 seconds
            raise ValueError("Audio file too short (minimum 0.1 seconds required)")
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        print(f"Audio normalized. Min: {np.min(audio):.3f}, Max: {np.max(audio):.3f}")
        
        # Extract MFCC features
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            features = np.mean(mfcc.T, axis=0)
            print(f"Successfully extracted MFCC features: {features.shape}")
            
            # Validate features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError("Invalid MFCC features (NaN or Inf values)")
            
            return features
            
        except Exception as e:
            print(f"MFCC extraction failed: {e}")
            raise Exception(f"Could not extract features from audio. Error: {str(e)}")
        
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        raise Exception(f"Audio processing failed: {str(e)}")

def get_file_info(file_path):
    """Get information about the audio file"""
    try:
        import mutagen
        from mutagen import File
        
        audio_file = File(file_path)
        if audio_file is not None:
            info = {
                'duration': getattr(audio_file.info, 'length', 0),
                'bitrate': getattr(audio_file.info, 'bitrate', 0),
                'sample_rate': getattr(audio_file.info, 'sample_rate', 0),
                'channels': getattr(audio_file.info, 'channels', 0)
            }
            print(f"File info: {info}")
            return info
    except ImportError:
        print("Mutagen not available for file info")
    except Exception as e:
        print(f"Could not get file info: {e}")
    
    return None

@app.route("/")
def index():
    return render_template("complete_project.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("=== Received prediction request ===")
        
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({"success": False, "error": "No audio file provided"})
        
        file = request.files["audio"]
        print(f"Received file: {file.filename}")
        print(f"Content type: {file.content_type}")
        print(f"File size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({"success": False, "error": "No file selected"})
        
        # Determine file extension
        file_extension = '.wav'  # default
        original_filename = file.filename or 'audio'
        
        if file.content_type:
            print(f"Content type: {file.content_type}")
            if 'webm' in file.content_type:
                file_extension = '.webm'
            elif 'mp4' in file.content_type or 'mp4a' in file.content_type:
                file_extension = '.mp4'
            elif 'mpeg' in file.content_type or 'mp3' in file.content_type:
                file_extension = '.mp3'
            elif 'wav' in file.content_type:
                file_extension = '.wav'
        
        # Also check filename extension
        if original_filename:
            name_lower = original_filename.lower()
            if name_lower.endswith('.webm'):
                file_extension = '.webm'
            elif name_lower.endswith('.mp4'):
                file_extension = '.mp4'
            elif name_lower.endswith('.mp3'):
                file_extension = '.mp3'
            elif name_lower.endswith('.wav'):
                file_extension = '.wav'
        
        print(f"Using file extension: {file_extension}")
        print(f"Original filename: {original_filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
            print(f"Saved temporary file: {temp_path}")
            print(f"Temp file size: {os.path.getsize(temp_path)} bytes")
        
        try:
            # Get file info if possible
            file_info = get_file_info(temp_path)
            
            # Extract features
            print("=== Extracting features ===")
            features = extract_features(temp_path)
            features = np.expand_dims(features, axis=0)
            print(f"Features shape: {features.shape}")

            # Make prediction
            print("=== Making prediction ===")
            prediction = model.predict(features, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotions[emotion_idx]
            confidence = float(np.max(prediction)) * 100
            
            print(f"Prediction: {emotion} ({confidence:.1f}%)")
            print(f"All probabilities: {prediction[0]}")
            
            # Get all probabilities
            probabilities = {
                'neutral': float(prediction[0][0]),
                'happy': float(prediction[0][1]),
                'sad': float(prediction[0][2]),
                'angry': float(prediction[0][3]),
                'fear': float(prediction[0][4]),
                'disgust': float(prediction[0][5]),
                'surprise': float(prediction[0][6])
            }

            result = {
                "success": True,
                "emotion": emotion,
                "confidence": f"{confidence:.1f}",
                "probabilities": probabilities
            }
            
            print("=== Prediction successful ===")
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
                
    except Exception as e:
        print(f"=== Prediction error ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
