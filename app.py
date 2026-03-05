import os
import numpy as np
import librosa
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Processor, Wav2Vec2ForCTC

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import threading

app = Flask(__name__)
# EMAIL CONFIGURATION (User must update these)
SENDER_EMAIL = "pavanibajjuri02@gmail.com"  # Placeholder
APP_PASSWORD = "gcihquvnstquzkcr"     # Placeholder
RECIPIENT_EMAIL = "pavaniavrr@gmail.com"
 # Placeholder
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

def send_email_alert(sound_class, danger_level, description, suggestion, attachment_path):
    """Sends an email alert in a background thread."""
    def _send():
        if "your_email" in SENDER_EMAIL:
            print("Email alert skipped: Placeholder credentials not updated.")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = f"SoundGuard Alert: {sound_class} Detected! [{danger_level}]"

            body = f""""
            SoundGuard Security Alert
            -------------------------
            Detected Sound: {sound_class}
            Danger Level: {danger_level}
            
            Scene Analysis:
            {description}
            
            Safety Suggestion:
            {suggestion}
            
            Audio file attached.
            """
            msg.attach(MIMEText(body, 'plain'))

            # Attach Audio File
            if os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(attachment_path)}",
                )
                msg.attach(part)

            # Send
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
            server.quit()
            print(f"Email alert sent to {RECIPIENT_EMAIL}")

        except Exception as e:
            print(f"Failed to send email alert: {e}")

    # Run in background thread to avoid blocking response
    thread = threading.Thread(target=_send)
    thread.start()

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
        return None
        
    return {
        "description": description,
        "primary_label": detected_sounds[0] if detected_sounds else None,
        "primary_score": scores[0].item() if detected_sounds else 0.0,
        "all_sounds": detected_sounds
    }

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
    # Verify models are loaded
    if rf_model is None:
         load_model() # Try loading again
         if rf_model is None:
              return jsonify({'error': 'Random Forest Model not loaded.'}), 500

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
        
        # ML Prediction (Random Forest)
        ml_pred_class = "Unknown"
        ml_confidence = 0.0
        try:
            # Reshape for single sample prediction (1, n_features)
            features = features.reshape(1, -1)
            
            # Check if model supports probability
            if hasattr(rf_model, "predict_proba"):
                probas = rf_model.predict_proba(features)[0]
                max_idx = np.argmax(probas)
                ml_pred_class = rf_model.classes_[max_idx]
                ml_confidence = probas[max_idx]
            else:
                ml_pred_class = rf_model.predict(features)[0]
                ml_confidence = 1.0 # Fallback if no proba
        except Exception as e:
             print(f"ML Prediction Error: {e}")

        # Perform Deep Learning Scene Analysis (AST + Wav2Vec2)
        scene_data = analyze_scene(filepath)
        
        # DECISION LOGIC: Prioritize Transformer (AST)
        final_class = ml_pred_class # Default to ML
        final_score = ml_confidence
        source = "ML (Random Forest)"
        
        if scene_data and scene_data.get("primary_label"):
            # AST detected something with high confidence?
            # Note: AST labels are from AudioSet (527 classes), ESC-50 has 50.
            # We display the AST label directly as it's more descriptive.
            final_class = scene_data["primary_label"]
            final_score = scene_data["primary_score"]
            source = "CNN"
        
        # Determine Danger Level (Map AST labels if possible, or fall back to ML mapping)
        # Since AST labels are different, we try to match substrings or use ML's danger level if AST is obscure
        danger_level = "UNKNOWN"
        
        # Simple heuristic: Check our known danger keywords in the AST label
        lower_class = final_class.lower()
        if any(x in lower_class for x in ['explosion', 'gun', 'siren', 'fire', 'alarm', 'scream', 'cry']):
             danger_level = 'DANGER'
        elif any(x in lower_class for x in ['engine', 'vehicle', 'thunder', 'storm', 'horn', 'train']):
             danger_level = 'WARNING'
        elif source == "ML (Random Forest)":
             danger_level = DANGER_LEVELS.get(final_class, 'SAFE')
        else:
             # Default assumption for unknown AST labels
             danger_level = 'SAFE'

        suggestion = SUGGESTIONS.get(danger_level, SUGGESTIONS['SAFE'])

        # Determine Hearing Safety
        safety_status = "Safe"
        if danger_level == 'DANGER':
            safety_status = "High Risk (Harmful)"
        elif danger_level == 'WARNING':
            safety_status = "Moderate Risk"
            
        # TRIGGER EMAIL ALERT if Danger/Warning
        if danger_level in ['DANGER', 'WARNING']:
            send_email_alert(
                sound_class=final_class,
                danger_level=danger_level,
                description=scene_data.get("description", "No description"),
                suggestion=suggestion,
                attachment_path=filepath
            )

        return jsonify({
            'primary': {
                'class': final_class,
                'score': f"{final_score*100:.1f}%",
                'source': source,
                'danger_level': danger_level,
                'hearing_safety': safety_status,
                'suggestion': suggestion
            },
            'secondary_ml': {
                'class': ml_pred_class,
                'score': f"{ml_confidence*100:.1f}%"
            },
            'scene_description': scene_data.get("description", "Analysis pending") if scene_data else "Analysis failed",
            'file_url': f"/static/uploads/{filename}"
        })
        


if __name__ == '__main__':
    print("Starting Flask Server...")
    load_model()
    # Disable reloader to prevent restart loops if temp files are modified
    app.run(debug=True, use_reloader=False)
