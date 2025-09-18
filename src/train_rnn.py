import os
import numpy as np
from src.model_utils import build_rnn_model, train_model
from src.utils import load_features_from_file, GENRES

def main():
    """
    Main function to train RNN model on MFCC + Chroma features.
    """
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    model_dir = os.path.join(BASE_DIR, 'models')

    # Load processed data
    print("Loading RNN training data...")
    X_train, y_train = load_features_from_file(os.path.join(processed_dir, 'rnn_train.npz'))
    X_val, y_val = load_features_from_file(os.path.join(processed_dir, 'rnn_val.npz'))

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {len(GENRES)}")

    # Build model
    input_shape = X_train.shape[1:]  # (time_steps, features)
    model = build_rnn_model(input_shape)
    print("RNN Model Summary:")
    model.summary()

    # Train model
    print("\nStarting RNN model training...")
    trained_model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        model_name='rnn_mfcc_chroma', model_dir=model_dir
    )

    print("RNN model training completed!")

    # Save training history
    history_path = os.path.join(model_dir, 'rnn_training_history.npy')
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()
