# 🎧 SoundGuard AI - Audio Event Detection & Safety Analysis

An intelligent web application that uses machine learning to detect, classify, and assess environmental sounds in real-time. Combines traditional ML with deep learning to provide comprehensive audio analysis including danger level classification, hearing safety assessment, speech recognition, and scene understanding.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### 🔊 Audio Analysis
- **Sound Classification**: Identifies 50+ environmental sound categories (ESC-50 dataset)
- **Multi-Label Detection**: Detects multiple overlapping sounds using Audio Spectrogram Transformer
- **Speech Recognition**: Transcribes speech content using Wav2Vec2
- **Scene Understanding**: Generates natural language descriptions of audio scenes

### ⚠️ Safety Assessment
- **Danger Level Classification**: Categorizes sounds into Safe/Warning/Danger levels
- **Hearing Safety Indicator**: Alerts users to potentially harmful noise exposure
- **Safety Suggestions**: Provides actionable recommendations based on detected sounds

### 🎵 Real-Time Features
- **Live Recording**: Browser-based microphone capture with WAV encoding
- **Frequency Visualization**: Animated real-time frequency spectrum display
- **Instant Analysis**: Sub-second processing and response time

### 🎨 Premium UI/UX
- **Dark Mode Design**: Modern glassmorphism aesthetic with smooth animations
- **Drag-and-Drop**: Intuitive file upload interface
- **Responsive Layout**: Works on desktop, tablet, and mobile devices

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Edge)

### Installation

1. **Clone or download the repository**
```bash
cd "d:\shiva\Sound detection"
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install flask librosa scikit-learn pandas numpy resampy torch transformers
```

4. **Train the model** (one-time setup)
```bash
python model_train.py
```
*Note: This will take 5-10 minutes. Requires the ESC-50 dataset in `dataset/audio/audio/`*

5. **Run the application**
```bash
python app.py
```

6. **Open in browser**
```
http://127.0.0.1:5000
```

---

## 📖 Usage

### Upload Audio File
1. Click **"Upload Audio"** tab
2. Drag and drop a `.wav`, `.mp3`, or `.ogg` file (or click "Choose File")
3. Click **"Analyze Sound"**
4. View results: sound class, danger level, hearing safety, scene description, and frequency visualization

### Record Audio
1. Click **"Record Audio"** tab
2. Click **"Start Recording"** (grant microphone permission if prompted)
3. Speak or capture ambient sound for a few seconds
4. Click **"Stop Recording"**
5. Click **"Analyze Sound"**

### Results Interpretation
- **Detected Sound**: Primary sound category (e.g., "dog_bark", "rain", "siren")
- **Danger Level**: 
  - 🟢 **SAFE**: Benign sounds (rain, typing, birds)
  - 🟡 **WARNING**: Potentially harmful with prolonged exposure (jackhammer, engine)
  - 🔴 **DANGER**: Immediate threats (explosion, gunshot, chainsaw)
- **Harmful to Hearing**: Risk assessment for hearing damage
- **Scene Analysis**: Natural language description combining sound events and speech
- **Suggestion**: Safety recommendations based on detected sound

---

## 🏗️ Project Structure

```
Sound detection/
├── app.py                      # Flask backend (main application)
├── model_train.py              # Random Forest training script
├── model.pkl                   # Trained Random Forest model (generated)
├── README.md                   # This file
├── dataset/                    # ESC-50 audio dataset
│   └── audio/audio/           # Audio files (.wav)
├── static/
│   ├── css/
│   │   └── style.css          # Premium dark theme styles
│   ├── js/
│   │   └── script.js          # Frontend logic & recording
│   └── uploads/               # Temporary audio storage (auto-created)
└── templates/
    └── index.html             # Main UI template
```

---

## 🧠 Technologies & Techniques

### Machine Learning
- **Random Forest Classifier** (scikit-learn): Fast CPU-based classification of 50 sound categories
- **Audio Spectrogram Transformer** (MIT/ast-finetuned-audioset): Deep learning for multi-label sound detection
- **Wav2Vec2** (facebook/wav2vec2-base-960h): Speech-to-text transcription

### Audio Processing
- **Librosa**: Feature extraction (MFCCs, Chroma, Spectral Contrast, Tonnetz)
- **Web Audio API**: Real-time microphone capture and frequency analysis

### Backend
- **Flask**: Lightweight Python web framework
- **PyTorch**: Deep learning inference
- **Transformers** (Hugging Face): Pre-trained model integration

### Frontend
- **HTML5/CSS3**: Responsive UI with glassmorphism design
- **Vanilla JavaScript**: No frameworks, optimized performance
- **Canvas API**: Real-time frequency visualization

---

## 📊 Model Details

### Random Forest (Primary Classifier)
- **Dataset**: ESC-50 (2000 samples, 50 classes)
- **Features**: 65-dimensional vector (13 MFCCs + Chroma + Spectral Contrast + Tonnetz)
- **Accuracy**: ~70-80% on test set
- **Inference Time**: 50-100ms (CPU)

### Deep Learning Models
- **AST**: 527 AudioSet classes, multi-label detection, confidence threshold 15%
- **Wav2Vec2**: Pre-trained on 960 hours of speech, CTC decoding

---

## 🔧 Configuration

### Danger Level Mapping
Edit `app.py` to customize sound categorization:
```python
DANGER_LEVELS = {
    'chainsaw': 'DANGER',
    'siren': 'DANGER',
    'dog_bark': 'WARNING',
    'rain': 'SAFE',
    # Add or modify mappings here
}
```

### Safety Suggestions
Edit `SUGGESTIONS` dictionary in `app.py` to customize recommendations.

---

## 🐛 Troubleshooting

### Model Not Loaded
- **Error**: "Model not loaded" or "Model file not found"
- **Solution**: Run `python model_train.py` to train the model first

### Deep Learning Models Not Loading
- **Error**: "Error loading deep learning models"
- **Solution**: 
  1. Check internet connection (models download on first run)
  2. Ensure `torch` and `transformers` are installed
  3. Wait 2-3 minutes for model download (~1GB)

### Audio Recording Error
- **Error**: "Microphone access denied"
- **Solution**: Grant browser microphone permissions, use HTTPS or localhost

### Feature Mismatch Error
- **Error**: "X has 65 features, but model is expecting 40"
- **Solution**: Retrain the model with current feature extraction logic

---

## 📝 Dataset

This project uses the **ESC-50 dataset**:
- **Source**: [ESC-50 on GitHub](https://github.com/karolpiczak/ESC-50)
- **Size**: 2000 audio clips (5 seconds each)
- **Classes**: 50 environmental sound categories
- **Format**: WAV (44.1 kHz, mono)

Download and place in: `dataset/audio/audio/`

---

## 🚧 Known Limitations

- Speech transcription accuracy depends on audio quality
- Model trained on 5-second clips; longer audio is truncated
- Deep learning models require ~2GB RAM
- First startup downloads models (~1GB)

---

## 🔮 Future Enhancements

- [ ] Export analysis results as JSON/CSV
- [ ] Sound event timestamp detection
- [ ] User authentication and history
- [ ] Mobile app (React Native)
- [ ] Integration with IoT devices
- [ ] Custom model fine-tuning interface

---

## 📄 License

MIT License - feel free to use for personal or commercial projects.

---

## 👨‍💻 Author

Developed as part of an audio event detection and safety analysis research project.

---

## 🙏 Acknowledgments

- **ESC-50 Dataset**: Karol J. Piczak
- **Audio Spectrogram Transformer**: MIT CSAIL
- **Wav2Vec2**: Facebook AI Research
- **Librosa**: Brian McFee et al.

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the project documentation
3. Verify dataset and model setup

---

**⭐ If you find this project useful, please consider giving it a star!**
