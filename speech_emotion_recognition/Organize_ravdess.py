import os, shutil 
RAVDESS_PATH = "audio_speech_actors_01-24/" 
DEST_PATH = "dataset/" emotion_map = {"01": "neutral", 
"03": "happy", "04": "sad", "05": "angry"} for e in 
emotion_map.values(): os.makedirs(os.path.join(DEST_PATH, 
e), exist_ok=True) 
counts = {e: 0 for e in 
emotion_map.values()} for root, dirs, 
files in os.walk(RAVDESS_PATH): for 
file in files: 
if file.endswith(".wav"): 

parts = 
file.split("-") if 
len(parts) > 2: 
emotion_code = parts[2] if 
emotion_code in 
emotion_map: label = 
emotion_map[emotion_code
] 
shutil.copy(os.path.join(root, file), os.path.join(DEST_PATH, label, file)) 
counts[label] += 1 print("Dataset 
reorganized into:", DEST_PATH) for 
emotion, count in counts.items(): 
print(f"{emotion}: {count} files")