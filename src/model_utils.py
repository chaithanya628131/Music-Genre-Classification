import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    LSTM, Bidirectional, BatchNormalization, Reshape
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def build_cnn_model(input_shape):
    """
    Build CNN model for spectrogram classification.

    Args:
        input_shape (tuple): Input shape for the model

    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = Sequential([
        # Reshape for CNN input (add channel dimension)
        Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),

        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_rnn_model(input_shape):
    """
    Build RNN model for MFCC + Chroma classification.

    Args:
        input_shape (tuple): Input shape for the model

    Returns:
        tf.keras.Model: Compiled RNN model
    """
    model = Sequential([
        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(32)),
        Dropout(0.3),

        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_callbacks(model_name, model_dir='../models'):
    """
    Get training callbacks for model training.

    Args:
        model_name (str): Name of the model
        model_dir (str): Directory to save models

    Returns:
        list: List of callbacks
    """
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, f'{model_name}_best.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    return [checkpoint, early_stopping]

def train_model(model, X_train, y_train, X_val, y_val, model_name, model_dir='../models'):
    """
    Train a model with given data.

    Args:
        model (tf.keras.Model): Model to train
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_val (np.array): Validation features
        y_val (np.array): Validation labels
        model_name (str): Name of the model
        model_dir (str): Directory to save models

    Returns:
        tf.keras.Model: Trained model
    """
    callbacks = get_callbacks(model_name, model_dir)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(model_dir, f'{model_name}_final.h5')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    return model, history

def evaluate_model(model, X_test, y_test, model_name, genres):
    """
    Evaluate model performance.

    Args:
        model (tf.keras.Model): Trained model
        X_test (np.array): Test features
        y_test (np.array): Test labels
        model_name (str): Name of the model
        genres (list): List of genre names

    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{model_name} Test Results:")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=genres))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=genres, yticklabels=genres)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # Ensure the directory exists before saving
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'loss': loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred_classes, target_names=genres, output_dict=True)
    }

def load_trained_model(model_path):
    """
    Load a trained model from file.

    Args:
        model_path (str): Path to the model file

    Returns:
        tf.keras.Model: Loaded model
    """
    return load_model(model_path)

def predict_genre(model, features, genres):
    """
    Predict genre for given features.

    Args:
        model (tf.keras.Model): Trained model
        features (np.array): Input features
        genres (list): List of genre names

    Returns:
        tuple: (predicted_genre, confidence)
    """
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]

    return genres[predicted_class], confidence
