import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import librosa
from src.utils import (
    load_audio_file, extract_mfcc, extract_chroma, extract_spectrogram,
    normalize_features, pad_or_truncate_features, GENRES
)
from src.model_utils import load_trained_model, predict_genre

# Flask app configuration
app = Flask(__name__, template_folder='../app/templates', static_folder='../app/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained models
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
cnn_model = None
rnn_model = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load trained models."""
    global cnn_model, rnn_model

    cnn_path = os.path.join(MODEL_DIR, 'cnn_spectrogram_best.h5')
    rnn_path = os.path.join(MODEL_DIR, 'rnn_mfcc_chroma_best.h5')

    try:
        if os.path.exists(cnn_path):
            cnn_model = load_trained_model(cnn_path)
            print("CNN model loaded successfully")
        else:
            print(f"CNN model not found at {cnn_path}")

        if os.path.exists(rnn_path):
            rnn_model = load_trained_model(rnn_path)
            print("RNN model loaded successfully")
        else:
            print(f"RNN model not found at {rnn_path}")

    except Exception as e:
        print(f"Error loading models: {e}")

def preprocess_audio_for_cnn(file_path):
    """Preprocess audio file for CNN model (spectrogram)."""
    audio, sr = load_audio_file(file_path)
    if audio is None:
        return None

    # Extract spectrogram
    spec = extract_spectrogram(audio, sr)

    # Normalize and pad/truncate
    spec_norm = normalize_features(spec)
    spec_padded = pad_or_truncate_features(spec_norm, 128)  # Match training shape

    # Add batch dimension
    return np.expand_dims(spec_padded, axis=0)

def preprocess_audio_for_rnn(file_path):
    """Preprocess audio file for RNN model (MFCC + Chroma)."""
    audio, sr = load_audio_file(file_path)
    if audio is None:
        return None

    # Extract MFCC and Chroma
    mfcc = extract_mfcc(audio, sr)
    chroma = extract_chroma(audio, sr)

    # Normalize
    mfcc_norm = normalize_features(mfcc)
    chroma_norm = normalize_features(chroma)

    # Pad/truncate
    mfcc_padded = pad_or_truncate_features(mfcc_norm, 1300)
    chroma_padded = pad_or_truncate_features(chroma_norm, 1300)

    # Fix dimension mismatch by trimming to minimum length
    min_len = min(mfcc_padded.shape[0], chroma_padded.shape[0])
    mfcc_trimmed = mfcc_padded[:min_len]
    chroma_trimmed = chroma_padded[:min_len]

    # Concatenate features
    combined = np.concatenate([mfcc_trimmed, chroma_trimmed], axis=1)

    # Add batch dimension
    return np.expand_dims(combined, axis=0)

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Preprocess for both models
            cnn_features = preprocess_audio_for_cnn(file_path)
            rnn_features = preprocess_audio_for_rnn(file_path)

            predictions = {}

            # CNN prediction
            if cnn_model is not None and cnn_features is not None:
                cnn_genre, cnn_confidence = predict_genre(cnn_model, cnn_features, GENRES)
                predictions['cnn'] = {
                    'genre': cnn_genre,
                    'confidence': round(cnn_confidence * 100, 2)
                }
            else:
                predictions['cnn'] = {'error': 'CNN model not available or preprocessing failed'}

            # RNN prediction
            if rnn_model is not None and rnn_features is not None:
                rnn_genre, rnn_confidence = predict_genre(rnn_model, rnn_features, GENRES)
                predictions['rnn'] = {
                    'genre': rnn_genre,
                    'confidence': round(rnn_confidence * 100, 2)
                }
            else:
                predictions['rnn'] = {'error': 'RNN model not available or preprocessing failed'}

            # Clean up uploaded file
            os.remove(file_path)

            return render_template('index.html', predictions=predictions, filename=filename)

        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('index.html', error=f"Error processing file: {str(e)}")

    return render_template('index.html', error="Invalid file type. Please upload a supported audio file.")

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return render_template('index.html', error="File too large. Maximum size is 16MB."), 413

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
