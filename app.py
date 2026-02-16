import os
import numpy as np
import librosa
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor, Wav2Vec2ForCTC

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload dir exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Models
MODEL_PATH = "model.pkl"
rf_model = None

# Deep Learning Models (Global)
ast_extractor = None
ast_model = None
speech_processor = None
speech_model = None

def load_deep_models():
    global ast_extractor, ast_model, speech_processor, speech_model
    try:
        print("Loading Deep Learning Models... This may take a moment.")
        # AST - Audio Spectrogram Transformer
        ast_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        ast_model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Wav2Vec2 - Speech Recognition
        speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        print("Deep Learning Models Loaded Successfully.")
    except Exception as e:
        print(f"Error loading deep learning models: {e}")

def analyze_scene(file_path):
    description = "Analysis failed."
    try:
        # Load audio for DL models (16kHz required)
        audio_input, sr = librosa.load(file_path, sr=16000)
        
        # 1. AST Scene Detection
        if ast_extractor and ast_model:
            inputs = ast_extractor(audio_input, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = ast_model(**inputs).logits
            
            probs = torch.sigmoid(logits)[0]
            # Get top predictions
            top_k = 3
            scores, indices = torch.topk(probs, top_k)
            
            detected_sounds = []
            for score, idx in zip(scores, indices):
                if score > 0.15: # Threshold
                    label = ast_model.config.id2label[idx.item()]
                    detected_sounds.append(label)
        
        # 2. Speech Transcription
        transcription = ""
        if speech_processor and speech_model:
            inputs = speech_processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = speech_model(**inputs).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = speech_processor.batch_decode(predicted_ids)[0].lower()

        # 3. Generate Description
        if transcription and len(transcription) > 2:
            description = f"Speech detected: \"{transcription}\"."
            if detected_sounds:
                bg_sounds = ", ".join([s for s in detected_sounds if "speech" not in s.lower()])
                if bg_sounds:
                    description += f" Background sounds include: {bg_sounds}."
        elif detected_sounds:
            main_sounds = ", ".join(detected_sounds)
            description = f"The audio contains sounds like: {main_sounds}."
        else:
            description = "No distinct events or speech detected."
            
    except Exception as e:
        print(f"Deep learning analysis error: {e}")
        description = "Could not analyze scene details."
        
    return description

# Danger Categories Mapping
DANGER_LEVELS = {
    # Danger
    'glass_breaking': 'DANGER',
    'gunshot': 'DANGER',
    'crying_baby': 'DANGER',  # Could be distress
    'siren': 'DANGER',
    'fireworks': 'DANGER', # Can be confused with gunshot
    'chainsaw': 'DANGER',
    'dog': 'WARNING', # Aggressive dog?
    'heliocopter': 'WARNING',
    
    # Warning
    'thunderstorm': 'WARNING',
    'engine': 'WARNING',
    'coughing': 'WARNING',
    'sneezing': 'WARNING',
    'car_horn': 'WARNING',
    'train': 'WARNING',
    'airplane': 'WARNING',
    
    # Safe / Neutral (Everything else defaults to Safe if not listed)
    'chirping_birds': 'SAFE',
    'rain': 'SAFE',
    'clapping': 'SAFE',
    'rooster': 'SAFE',
    'sea_waves': 'SAFE',
    'crickets': 'SAFE',
    'frog': 'SAFE',
    'cat': 'SAFE',
    'hen': 'SAFE',
    'pig': 'SAFE',
    'sheep': 'SAFE',
    'cow': 'SAFE',
    'brushing_teeth': 'SAFE',
    'keyboard_typing': 'SAFE',
    'footsteps': 'SAFE'
}

SUGGESTIONS = {
    'DANGER': "Immediate action recommended! Move away from the source or seek safety.",
    'WARNING': "Be cautious. Pay attention to your surroundings.",
    'SAFE': "Environment appears safe. You are good to stay."
}

def load_model():
    global rf_model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        print("Random Forest Model loaded successfully.")
    else:
        print("Model file not found. Please train the model first.")
    
    # Load DL models
    load_deep_models()

def extract_features(file_path):
    try:
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
        return np.hstack([mfccs_mean, chroma, contrast, tonnetz]), None
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, str(e) # Return tuple with error message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None:
        load_model()
        if rf_model is None:
             return jsonify({'error': 'Random Forest Model not loaded. Train it first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Feature Extraction
        features, error_msg = extract_features(filepath)
        if features is None:
             return jsonify({'error': f'Could not process audio file: {error_msg}'}), 400
        
        # Predict
        features = features.reshape(1, -1)
        try:
            prediction_index = rf_model.predict(features)[0] 
        except ValueError as e:
            print(f"Prediction Error: {e}")
            return jsonify({'error': 'Model incompatibility detected. The model is still training. Please wait for training to complete and then restart the app.'}), 503
        except Exception as e:
            print(f"Unexpected Prediction Error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500 # Check if model predicts index or label
        # If model was trained on strings, prediction is string.
        # In model_train.py, I used row['category'] which is string.
        predicted_class = prediction_index
        
        # Determine Danger Level
        danger_level = DANGER_LEVELS.get(predicted_class, 'SAFE')
        suggestion = SUGGESTIONS.get(danger_level, SUGGESTIONS['SAFE'])

        # Determine Hearing Safety
        safety_status = "Safe"
        if danger_level == 'DANGER':
            safety_status = "High Risk (Harmful)"
        elif danger_level == 'WARNING':
            safety_status = "Moderate Risk"
            
        # Perform Deep Learning Scene Analysis
        scene_description = analyze_scene(filepath)

        return jsonify({
            'class': predicted_class,
            'danger_level': danger_level,
            'hearing_safety': safety_status,
            'scene_description': scene_description,
            'suggestion': suggestion,
            'file_url': f"/static/uploads/{filename}"
        })

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
