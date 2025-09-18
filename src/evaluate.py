import os
import numpy as np
from src.model_utils import load_trained_model, evaluate_model
from src.utils import load_features_from_file, GENRES

def main():
    """
    Main function to evaluate trained models on test data.
    """
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    model_dir = os.path.join(BASE_DIR, 'models')

    # Load test data
    print("Loading test data...")
    X_cnn_test, y_cnn_test = load_features_from_file(os.path.join(processed_dir, 'cnn_test.npz'))
    X_rnn_test, y_rnn_test = load_features_from_file(os.path.join(processed_dir, 'rnn_test.npz'))

    print(f"CNN test data shape: {X_cnn_test.shape}")
    print(f"RNN test data shape: {X_rnn_test.shape}")

    # Evaluate CNN model
    print("\n" + "="*50)
    print("Evaluating CNN Model")
    print("="*50)

    cnn_model_path = os.path.join(model_dir, 'cnn_spectrogram_best.h5')
    if os.path.exists(cnn_model_path):
        cnn_model = load_trained_model(cnn_model_path)
        cnn_metrics = evaluate_model(cnn_model, X_cnn_test, y_cnn_test, 'CNN_Spectrogram', GENRES)
    else:
        print(f"CNN model not found at {cnn_model_path}")
        cnn_metrics = None

    # Evaluate RNN model
    print("\n" + "="*50)
    print("Evaluating RNN Model")
    print("="*50)

    rnn_model_path = os.path.join(model_dir, 'rnn_mfcc_chroma_best.h5')
    if os.path.exists(rnn_model_path):
        rnn_model = load_trained_model(rnn_model_path)
        rnn_metrics = evaluate_model(rnn_model, X_rnn_test, y_rnn_test, 'RNN_MFCC_Chroma', GENRES)
    else:
        print(f"RNN model not found at {rnn_model_path}")
        rnn_metrics = None

    # Compare models
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)

    if cnn_metrics and rnn_metrics:
        print(f"CNN Accuracy: {cnn_metrics['accuracy']:.4f}")
        print(f"RNN Accuracy: {rnn_metrics['accuracy']:.4f}")

        if cnn_metrics['accuracy'] > rnn_metrics['accuracy']:
            print("CNN model performs better!")
        elif rnn_metrics['accuracy'] > cnn_metrics['accuracy']:
            print("RNN model performs better!")
        else:
            print("Both models have similar performance!")

        # Save comparison results
        comparison = {
            'cnn_accuracy': cnn_metrics['accuracy'],
            'rnn_accuracy': rnn_metrics['accuracy'],
            'cnn_loss': cnn_metrics['loss'],
            'rnn_loss': rnn_metrics['loss']
        }

        np.save(os.path.join(model_dir, 'model_comparison.npy'), comparison)
        print("Model comparison saved!")

    else:
        print("Cannot compare models - one or both models not found")

if __name__ == "__main__":
    main()
