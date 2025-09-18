import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# GTZAN genres
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_audio_file(file_path, sr=22050, duration=30):
    """
    Load an audio file and return the audio data and sample rate.

    Args:
        file_path (str): Path to the audio file
        sr (int): Target sample rate
        duration (int): Duration to load in seconds

    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_features(audio, sr, n_mfcc=13):
    """
    Extract audio features: MFCCs, chroma, and spectrogram.

    Args:
        audio (np.array): Audio data
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients

    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features['mfccs'] = mfccs

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma'] = chroma

    # Spectrogram (Mel spectrogram)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['spectrogram'] = mel_spec_db

    return features

def normalize_features(features):
    """
    Normalize features to zero mean and unit variance.
    Works for both numpy arrays and dicts of arrays.
    """
    if isinstance(features, dict):  # case: dict of feature arrays
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                mean = np.mean(feature)
                std = np.std(feature) if np.std(feature) != 0 else 1e-8
                features[key] = (feature - mean) / std
        return features

    elif isinstance(features, np.ndarray):  # case: numpy array
        mean = np.mean(features)
        std = np.std(features) if np.std(features) != 0 else 1e-8
        return (features - mean) / std

    else:
        raise TypeError(f"Unsupported feature type: {type(features)}")

def pad_or_truncate(feature, target_length):
    """
    Pad or truncate feature to target length along time axis.

    Args:
        feature (np.array): Feature array with shape (n_features, time_steps) or (time_steps, n_features)
        target_length (int): Target length for time_steps

    Returns:
        np.array: Padded or truncated feature array
    """
    if feature.ndim == 3:
        # Assume (batch, time_steps, n_features)
        time_steps = feature.shape[1]
        if time_steps > target_length:
            return feature[:, :target_length, :]
        elif time_steps < target_length:
            pad_width = target_length - time_steps
            pad_shape = ((0, 0), (0, pad_width), (0, 0))
            return np.pad(feature, pad_shape, mode='constant')
        else:
            return feature
    else:
        # Assume (n_features, time_steps) for librosa features
        time_steps = feature.shape[1]
        if time_steps > target_length:
            return feature[:, :target_length]
        elif time_steps < target_length:
            pad_width = target_length - time_steps
            pad_shape = ((0, 0), (0, pad_width))
            return np.pad(feature, pad_shape, mode='constant')
        else:
            return feature


# Removed duplicate function - keeping the more comprehensive version below


def prepare_dataset(data_dir, save_path=None):
    """
    Prepare the dataset by extracting features from all audio files.

    Args:
        data_dir (str): Directory containing genre subdirectories
        save_path (str): Path to save processed data (optional)

    Returns:
        tuple: (X, y, label_encoder)
    """
    X = []
    y = []

    for genre in GENRES:
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.exists(genre_dir):
            print(f"Genre directory {genre_dir} not found")
            continue

        print(f"Processing genre: {genre}")
        for file in os.listdir(genre_dir):
            if file.endswith('.au'):
                file_path = os.path.join(genre_dir, file)
                audio, sr = load_audio_file(file_path)
                if audio is not None:
                    features = extract_features(audio, sr)
                    normalized_features = normalize_features(features)

                    # Combine features into a single array
                    combined = np.concatenate([
                        normalized_features['mfccs'],
                        normalized_features['chroma'],
                        normalized_features['spectrogram']
                    ], axis=0)

                    # Pad/truncate to fixed length (assuming 30s at 22050Hz = 66150 samples)
                    # For features, we need to calculate appropriate length
                    target_length = 1300  # Approximate for 30s audio
                    combined = pad_or_truncate(combined, target_length)

                    X.append(combined.T)  # Transpose for time-major
                    y.append(genre)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X = np.array(X)
    y_encoded = np.array(y_encoded)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((X, y_encoded, label_encoder), f)
        print(f"Dataset saved to {save_path}")

    return X, y_encoded, label_encoder

def split_dataset(X, y, test_size=0.2, val_size=0.2):
    """
    Split dataset into train, validation, and test sets.

    Args:
        X (np.array): Features
        y (np.array): Labels
        test_size (float): Test set proportion
        val_size (float): Validation set proportion

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_processed_data(file_path):
    """
    Load processed dataset from pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        tuple: (X, y, label_encoder)
    """
    with open(file_path, 'rb') as f:
        X, y, label_encoder = pickle.load(f)
    return X, y, label_encoder

def extract_mfcc(audio, sr, n_mfcc=13):
    """
    Extract MFCC features from audio.

    Args:
        audio (np.array): Audio data
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients

    Returns:
        np.array: MFCC features
    """
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

def extract_chroma(audio, sr):
    """
    Extract chroma features from audio.

    Args:
        audio (np.array): Audio data
        sr (int): Sample rate

    Returns:
        np.array: Chroma features
    """
    return librosa.feature.chroma_stft(y=audio, sr=sr)

def extract_spectrogram(audio, sr):
    """
    Extract spectrogram features from audio.

    Args:
        audio (np.array): Audio data
        sr (int): Sample rate

    Returns:
        np.array: Spectrogram features
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    return librosa.power_to_db(mel_spec, ref=np.max)

def pad_or_truncate_features(feature, target_length):
    """
    Pad or truncate feature to target length.

    Args:
        feature (np.array): Feature array
        target_length (int): Target length

    Returns:
        np.array: Padded or truncated feature
    """
    return pad_or_truncate(feature, target_length)

def encode_labels(y):
    """
    Encode string labels to integers.

    Args:
        y (np.array): String labels

    Returns:
        tuple: (encoded_labels, label_encoder)
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder

def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Split dataset into train, validation, and test sets.

    Args:
        X (np.array): Features
        y (np.array): Labels
        test_size (float): Test set proportion
        val_size (float): Validation set proportion

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    return split_dataset(X, y, test_size, val_size)

def save_features_to_file(X, y, file_path):
    """
    Save features and labels to a compressed numpy file.

    Args:
        X (np.array): Features
        y (np.array): Labels
        file_path (str): Path to save file
    """
    np.savez(file_path, X=X, y=y)

def load_features_from_file(file_path):
    """
    Load features and labels from a compressed numpy file.

    Args:
        file_path (str): Path to the file

    Returns:
        tuple: (X, y)
    """
    data = np.load(file_path)
    return data['X'], data['y']
