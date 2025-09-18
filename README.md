# Music Genre Classification

A machine learning project for classifying music genres using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) on the GTZAN dataset. The project includes data preprocessing, model training, evaluation, and a web application for real-time genre prediction from uploaded audio files.

## Features

- **Audio Feature Extraction**: Extracts MFCC, Chroma, and Spectrogram features from audio files using Librosa.
- **Dual Model Architecture**: Trains both CNN (on spectrograms) and RNN (on MFCC + Chroma) models for genre classification.
- **Data Preprocessing**: Normalizes features, handles padding/truncation, and splits data into training, validation, and test sets.
- **Web Application**: Flask-based web app for uploading audio files and getting predictions from both models.
- **Model Evaluation**: Includes confusion matrices and performance metrics for both models.
- **Support for Multiple Audio Formats**: Handles WAV, MP3, FLAC, and OGG files.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd music-genre-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the GTZAN dataset is placed in the `data/genre/` directory with subfolders for each genre (e.g., `blues/`, `classical/`, etc.).

## Usage

### Preprocessing

Run the preprocessing script to extract features and prepare the dataset:

```bash
python src/preprocess.py
```

This will:
- Load audio files from `data/genre/`
- Extract MFCC, Chroma, and Spectrogram features
- Normalize and pad/truncate features to fixed lengths
- Split data into training (60%), validation (20%), and test (20%) sets
- Save processed data to `data/processed/` as NPZ files

### Training

Train the CNN model on spectrogram features:

```bash
python src/train_cnn.py
```

Train the RNN model on MFCC + Chroma features:

```bash
python src/train_rnn.py
```

Trained models and training history will be saved to the `models/` directory.

### Evaluation

Evaluate the trained models:

```bash
python src/evaluate.py
```

This will load the test data, make predictions, and generate confusion matrices saved as PNG files in the `docs/` directory.

### Running the Web App

Start the Flask web application:

```bash
python app/app.py
```

The app will run on `http://localhost:5000`. Upload an audio file to get genre predictions from both CNN and RNN models, along with confidence scores.

## Dataset

The project uses the GTZAN dataset, which contains 1000 audio files (30 seconds each) across 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

Raw audio files should be placed in `data/genre/` with one subfolder per genre. Processed features are saved in `data/processed/` as NPZ files for efficient loading.

## Models

- **CNN Model**: Trained on spectrogram features. Architecture includes convolutional layers, max pooling, and dense layers.
- **RNN Model**: Trained on combined MFCC and Chroma features. Uses LSTM layers for sequence modeling.

Both models are saved as HDF5 files in the `models/` directory. Training history is also saved for analysis.

## Results

Model performance can be evaluated using the confusion matrices generated in `docs/`:
- `CNN_Spectrogram_confusion_matrix.png`
- `RNN_MFCC_Chroma_confusion_matrix.png`

These show the classification accuracy across different genres for both models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
