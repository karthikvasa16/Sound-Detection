import os
import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Configuration
DATASET_DIR = r"d:\shiva\Sound detection\dataset"
AUDIO_DIR = os.path.join(DATASET_DIR, "audio", "audio")
CSV_PATH = os.path.join(DATASET_DIR, "esc50.csv")
MODEL_PATH = "model.pkl"

def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        
        # Spectral Contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        
        # Tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)
        
        # Combine
        return np.hstack([mfccs_mean, chroma, contrast, tonnetz])
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None

def train_model():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    
    features = []
    labels = []
    
    print("Extracting features (this may take a few minutes)...")
    # Iterate through metadata and load audio files
    total_files = len(df)
    for index, row in df.iterrows():
        file_name = os.path.join(AUDIO_DIR, row['filename'])
        class_label = row['category']
        
        if index % 100 == 0:
            print(f"Processing {index}/{total_files}...")
            
        data = extract_features(file_name)
        if data is not None:
            features.append(data)
            labels.append(class_label)

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
