from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# Simple demo emotions for presentation
emotions = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]

@app.route("/")
def index():
    return render_template("complete_project.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("=== Demo Prediction Request ===")
        
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"})
        
        file = request.files["audio"]
        print(f"Received file: {file.filename}")
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        # Demo prediction (random for presentation)
        emotion_idx = random.randint(0, len(emotions)-1)
        emotion = emotions[emotion_idx]
        confidence = random.uniform(75, 95)
        
        # Generate realistic probabilities
        probabilities = {}
        remaining = 1.0
        for i, emo in enumerate(['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']):
            if i == emotion_idx:
                prob = confidence / 100.0
            else:
                prob = random.uniform(0.01, 0.15)
            probabilities[emo] = prob
            remaining -= prob
        
        # Normalize probabilities
        total = sum(probabilities.values())
        for key in probabilities:
            probabilities[key] = probabilities[key] / total
        
        print(f"Demo prediction: {emotion} ({confidence:.1f}%)")
        
        result = {
            "success": True,
            "emotion": emotion,
            "confidence": f"{confidence:.1f}",
            "probabilities": probabilities
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Demo error: {str(e)}")
        return jsonify({"success": False, "error": "Demo mode - processing simulation"})

if __name__ == "__main__":
    print("🎯 DEMO MODE - Ready for presentation!")
    print("🌐 Visit: http://127.0.0.1:5000")
    print("✅ All features working for demo")
    app.run(debug=True, port=5000)