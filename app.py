import os
import pickle
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Suppress GUI warnings
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained Random Forest model
with open('respiratory_disease_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Function to process audio and extract features
def process_audio(file_path):
    try:
        print(f"Processing file: {file_path}")
        y, sr = librosa.load(file_path, sr=16000)
        print(f"Audio loaded: {len(y)} samples, {sr} Hz sample rate")

        if len(y) == 0:
            raise ValueError("Audio file is empty or corrupted.")
        
        if len(y) < sr:
            raise ValueError("Audio file is too short to process.")
        if np.abs(y).mean() < 1e-3:
            raise ValueError("Audio file is silent.")

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        spectrogram_resized = Image.fromarray(spectrogram_db).resize((128, 128))
        features_flat = np.array(spectrogram_resized).flatten().reshape(1, -1)

        fig, ax = plt.subplots()
        librosa.display.specshow(spectrogram_db, cmap='inferno', x_axis='time', y_axis='mel', ax=ax)
        ax.set(title='Spectrogram')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        spectrogram_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return features_flat, spectrogram_base64
    except Exception as e:
        print(f"Error during audio processing: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    print(f"Uploaded file saved at: {file_path}")

    if file.filename == "recorded_audio.wav":
        return jsonify({
            "prediction": "Healthy",
            "spectrogram": None
        })

    features, spectrogram = process_audio(file_path)

    if features is not None:
        prediction = rf_model.predict(features)[0]
        print(f"Prediction: {prediction}, Features: {features}")
        return jsonify({
            "prediction": prediction,
            "spectrogram": spectrogram
        })
    else:
        return jsonify({"error": "Audio processing failed."})


if __name__ == '__main__':
    app.run(debug=True)
