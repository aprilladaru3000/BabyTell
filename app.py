import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app) 

SAMPLE_RATE = 22050
DURATION = 2
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512

model = None
label_encoder = None
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_PATH = os.path.join(BASE_DIR, "baby_cry_model.h5")
    ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

    print("--- Attempting to load resources ---")
    print(f"Base directory: {BASE_DIR}")
    
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found. Please ensure 'baby_cry_model.h5' is in the same folder as app.py.")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Loading encoder from: {ENCODER_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder file not found. Please ensure 'label_encoder.pkl' is in the same folder as app.py.")
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
        
    print("--- Model and label encoder loaded successfully. ---")

except Exception as e:
    print(f"\n!!! FATAL ERROR loading model or encoder: {e} !!!")
    print("Please make sure 'baby_cry_model.h5' and 'label_encoder.pkl' are in the same directory as this script.")


def preprocess_audio(file_path, target_samples=SAMPLES_PER_TRACK):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        if len(signal) > target_samples:
            signal = signal[:target_samples]
        else:
            padding = target_samples - len(signal)
            signal = np.pad(signal, (0, padding), mode='constant')

        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
        
        return mel_spectrogram
    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    return "<h1>Baby Cry Classification API</h1><p>Send a POST request to /predict with a .wav file.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoder is None:
        return jsonify({"error": "Model is not loaded. Check server logs for the fatal error."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found in the request."}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        mel_spec = preprocess_audio(file)
        if mel_spec is None:
            return jsonify({"error": "Failed to process audio file."}), 500

        mel_spec = np.expand_dims(mel_spec, axis=0)
        mel_spec = np.expand_dims(mel_spec, axis=-1)

        prediction = model.predict(mel_spec)
        
        predicted_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        return jsonify({
            "prediction": predicted_label,
            "confidence": f"{confidence:.2%}"
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
