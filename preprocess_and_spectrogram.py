import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Dataset Directory
data_dir = r'D:\OneDrive\Desktop\today\audio'

# Define labels and corresponding prefixes
labels = {
    'copd': 'copd',
    'asthama': 'asthama',
    'pneumonia': 'pneumonia'
}

# Function to extract features
def extract_features_from_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load the audio file
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        # Resize spectrogram to 128x128
        spectrogram_resized = Image.fromarray(spectrogram_db).resize((128, 128))
        spectrogram_array = np.array(spectrogram_resized)
        return spectrogram_array.flatten()  # Flatten for Random Forest
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Function to extract features from the entire dataset
def extract_features_from_dataset():
    data = []
    target = []
    for label, prefix in labels.items():
        print(f"Processing {label} files...")
        for file_name in os.listdir(data_dir):
            if file_name.lower().startswith(prefix.lower()) and file_name.endswith('.wav'):
                file_path = os.path.join(data_dir, file_name)
                features = extract_features_from_file(file_path)
                if features is not None:
                    data.append(features)
                    target.append(label)
    return np.array(data), np.array(target)

# Extract features and labels
X, y = extract_features_from_dataset()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
with open('respiratory_disease_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

