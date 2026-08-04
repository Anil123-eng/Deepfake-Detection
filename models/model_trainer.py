"""
Model trainer module.

Implements the training pipeline for the hybrid CNN + RNN model using
TensorFlow/Keras, including callbacks for monitoring, early stopping,
learning-rate scheduling, and model checkpointing.
"""

import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from config import Config
from models.hybrid_model import HybridDeepfakeModel

logger = logging.getLogger(__name__)


class HybridModelTrainer:
    """
    Encapsulates the training/validation/test workflow for the hybrid model.
    """

    def __init__(self, model=None, config=None):
        self.config = config or Config
        self.model = model or self._create_default_model()
        self.history = None

    def _create_default_model(self):
        """Create a default hybrid model from config."""
        model_obj = HybridDeepfakeModel(
            sequence_length=self.config.SEQUENCE_LENGTH,
            frame_size=self.config.FRAME_SIZE,
            num_classes=self.config.NUM_CLASSES,
            backbone=self.config.CNN_BACKBONE,
            cnn_trainable=self.config.CNN_TRAINABLE,
            rnn_units=self.config.RNN_UNITS,
            rnn_type="lstm",
            dropout=self.config.RNN_DROPOUT,
        )
        model_obj.compile(learning_rate=self.config.LEARNING_RATE)
        return model_obj

    def _build_callbacks(self):
        """Create training callbacks for monitoring and saving."""
        self.config.ensure_directories()
        return [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.config.MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            callbacks.TensorBoard(
                log_dir=str(self.config.TRAINED_DIR / "logs"),
                histogram_freq=1,
            ),
        ]

    def train(self, X_train, y_train, X_val, y_val,
              batch_size=None, epochs=None):
        """
        Train the hybrid model.

        Args:
            X_train: Training sequences (samples, time, h, w, c).
            y_train: One-hot labels for training.
            X_val: Validation sequences.
            y_val: One-hot labels for validation.
            batch_size: Batch size override.
            epochs: Number of epochs override.
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        epochs = epochs or self.config.EPOCHS

        callbacks_list = self._build_callbacks()

        logger.info("Starting hybrid model training...")
        self.history = self.model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
        )
        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate the trained model on a test set."""
        loss, acc = self.model.model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
        return {"loss": float(loss), "accuracy": float(acc)}

    def predict(self, sequences):
        """Predict class probabilities for video sequences."""
        probs = self.model.model.predict(sequences, verbose=0)
        return probs

    def save(self, path=None):
        """Save the trained model and label mapping."""
        path = path or self.config.MODEL_PATH
        self.config.ensure_directories()
        self.model.save(path)
        # Save label mapping (0 -> REAL, 1 -> FAKE)
        labels = {"0": "REAL", "1": "FAKE"}
        with open(self.config.LABEL_ENCODER_PATH, "w") as f:
            json.dump(labels, f, indent=2)
        logger.info(f"Model saved to {path}")
        logger.info(f"Labels saved to {self.config.LABEL_ENCODER_PATH}")
        return path

    @classmethod
    def load(cls, path=None):
        """Load a trained trainer from disk."""
        path = path or Config.MODEL_PATH
        model = HybridDeepfakeModel.load(path)
        trainer = cls(model=model)
        return trainer
