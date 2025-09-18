import os
import numpy as np
from tqdm import tqdm
from src.utils import (
    GENRES, load_audio_file, extract_mfcc, extract_chroma, extract_spectrogram,
    normalize_features, pad_or_truncate_features, encode_labels, split_data,
    save_features_to_file
)

# Configuration
DATA_DIR = DATA_DIR = r"C:\Users\chait\OneDrive\Documents\Project\Music Genre Classification\data\genre"
  # GTZAN dataset directory
PROCESSED_DIR = '../data/processed'
MAX_LENGTH_MFCC = 1300  # Approximate length for 30s audio at 22050Hz with hop_length=512
MAX_LENGTH_SPEC = 128   # For spectrogram

def process_genre_files(genre, data_dir):
    """
    Process all audio files for a specific genre.

    Args:
        genre (str): Genre name
        data_dir (str): Path to data directory

    Returns:
        tuple: (mfcc_features, chroma_features, spec_features, labels)
    """
    genre_dir = os.path.join(data_dir, genre)
    mfcc_list = []
    chroma_list = []
    spec_list = []
    labels = []

    if not os.path.exists(genre_dir):
        print(f"Warning: Genre directory {genre_dir} not found")
        return [], [], [], []

    files = [f for f in os.listdir(genre_dir) if f.endswith('.au')]
    print(f"Processing {len(files)} files for genre: {genre}")

    for file in tqdm(files, desc=f"Processing {genre}"):
        file_path = os.path.join(genre_dir, file)
        audio, sr = load_audio_file(file_path)

        if audio is None:
            continue

        # Extract features
        mfcc = extract_mfcc(audio, sr)
        chroma = extract_chroma(audio, sr)
        spec = extract_spectrogram(audio, sr)

        # Pad/truncate to fixed length
        mfcc = pad_or_truncate_features(mfcc, MAX_LENGTH_MFCC)
        chroma = pad_or_truncate_features(chroma, MAX_LENGTH_MFCC)
        spec = pad_or_truncate_features(spec, MAX_LENGTH_SPEC)

        mfcc_list.append(mfcc)
        chroma_list.append(chroma)
        spec_list.append(spec)
        labels.append(genre)

    return mfcc_list, chroma_list, spec_list, labels

def preprocess_dataset(data_dir, processed_dir):
    """
    Preprocess the entire GTZAN dataset.

    Args:
        data_dir (str): Path to raw data directory
        processed_dir (str): Path to save processed data
    """
    os.makedirs(processed_dir, exist_ok=True)

    all_mfcc = []
    all_chroma = []
    all_spec = []
    all_labels = []

    print("Starting dataset preprocessing...")

    for genre in GENRES:
        mfcc_list, chroma_list, spec_list, labels = process_genre_files(genre, data_dir)
        all_mfcc.extend(mfcc_list)
        all_chroma.extend(chroma_list)
        all_spec.extend(spec_list)
        all_labels.extend(labels)

    print(f"Total files processed: {len(all_labels)}")

    if len(all_labels) == 0:
        print("No audio files found. Please ensure the dataset is properly placed in the data directory.")
        return

    # Convert to numpy arrays
    X_mfcc = np.array(all_mfcc)
    X_chroma = np.array(all_chroma)
    X_spec = np.array(all_spec)
    y = np.array(all_labels)

    print(f"MFCC shape: {X_mfcc.shape}")
    print(f"Chroma shape: {X_chroma.shape}")
    print(f"Spectrogram shape: {X_spec.shape}")
    print(f"Labels shape: {y.shape}")

       # Normalize features
    print("Normalizing features...")
    X_mfcc_norm = np.array([normalize_features(track) for track in X_mfcc])
    X_chroma_norm = np.array([normalize_features(track) for track in X_chroma])
    X_spec_norm = np.array([normalize_features(track) for track in X_spec])

    # Align MFCC + Chroma time steps before concatenation
    print("Aligning MFCC and Chroma time steps...")
    min_time = min(X_mfcc_norm.shape[1], X_chroma_norm.shape[1])
    X_mfcc_norm = X_mfcc_norm[:, :min_time, :]
    X_chroma_norm = X_chroma_norm[:, :min_time, :]

    # Encode labels
    y_encoded, label_encoder = encode_labels(y)

    # Save label encoder
    np.save(os.path.join(processed_dir, 'label_encoder.npy'), label_encoder.classes_)

    # Split data for MFCC + Chroma (for RNN)
    print("Splitting data for RNN features...")
    X_rnn = np.concatenate([X_mfcc_norm, X_chroma_norm], axis=2)  # (samples, time, 25)
    X_rnn_train, X_rnn_val, X_rnn_test, y_train, y_val, y_test = split_data(
        X_rnn, y_encoded, test_size=0.2, val_size=0.2
    )

    # Save RNN data
    save_features_to_file(X_rnn_train, y_train, os.path.join(processed_dir, 'rnn_train.npz'))
    save_features_to_file(X_rnn_val, y_val, os.path.join(processed_dir, 'rnn_val.npz'))
    save_features_to_file(X_rnn_test, y_test, os.path.join(processed_dir, 'rnn_test.npz'))

    # Split data for Spectrogram (for CNN)
    print("Splitting data for CNN features...")
    X_cnn_train, X_cnn_val, X_cnn_test, y_train_cnn, y_val_cnn, y_test_cnn = split_data(
        X_spec_norm, y_encoded, test_size=0.2, val_size=0.2
    )

    # Save CNN data
    save_features_to_file(X_cnn_train, y_train_cnn, os.path.join(processed_dir, 'cnn_train.npz'))
    save_features_to_file(X_cnn_val, y_val_cnn, os.path.join(processed_dir, 'cnn_val.npz'))
    save_features_to_file(X_cnn_test, y_test_cnn, os.path.join(processed_dir, 'cnn_test.npz'))

    print("Preprocessing completed!")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

if __name__ == "__main__":
    preprocess_dataset(DATA_DIR, PROCESSED_DIR)
