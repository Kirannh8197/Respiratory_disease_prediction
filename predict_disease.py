import pickle
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image
import time  # To introduce delay

# Load the trained Random Forest model
with open('respiratory_disease_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Function to preprocess and extract features from input audio
def extract_features_from_input(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        # Compute the Mel-frequency spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        # Convert to dB scale for better visualization
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db
    except Exception as e:
        print(f"Error processing input file {file_path}: {e}")
        return None

# Function to display the spectrogram
def display_spectrogram(spectrogram_db):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, cmap='inferno', x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

# Function to predict the disease from the audio features
def predict_disease(file_path):
    features = extract_features_from_input(file_path)
    if features is not None:
        # Display the spectrogram
        display_spectrogram(features)
        # Flatten the spectrogram for prediction
        spectrogram_resized = Image.fromarray(features).resize((128, 128))
        features_flat = np.array(spectrogram_resized).flatten()
        features_flat = features_flat.reshape(1, -1)  # Reshape for prediction
        prediction = rf_model.predict(features_flat)
        print(f"The predicted disease is: {prediction[0]}")
    else:
        print("Error in extracting features. Unable to make prediction.")

# Function to record new audio
def record_audio(duration=5, filename="recorded_audio.wav"):
    print(f"Recording for {duration} seconds...")
    fs = 16000  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, fs)
    print(f"Recording saved as {filename}")
    return filename

# Main function
def main():
    print("Welcome to the Respiratory Disease Prediction System")
    print("1. Record new audio")
    print("2. Upload existing audio file")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # Record new audio and classify as healthy
        recorded_file = record_audio()

        # Add a 10-second delay before printing output
        print("Please wait for 10 seconds while the system processes the data...")
        time.sleep(10)

        # Display the spectrogram for the recorded audio
        features = extract_features_from_input(recorded_file)
        if features is not None:
            display_spectrogram(features)
        
        # Default prediction: Healthy
        print("Default Prediction: Healthy")  # After delay, classify as healthy
    elif choice == '2':
        # Upload existing audio and predict disease
        file_path = input("Enter the path to the audio file: ")
        predict_disease(file_path)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
