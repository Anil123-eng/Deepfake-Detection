"""
Hybrid CNN + RNN model module.

This module combines the CNN feature extractor and the RNN temporal model
into a single end-to-end deep learning architecture for deepfake detection.
It supports both TimeDistributed training (end-to-end) and feature-based
training (pre-extracted features fed to RNN).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from models.cnn_model import CNNFeatureExtractor
from models.rnn_model import RNNSequenceModel


class HybridDeepfakeModel:
    """
    Hybrid CNN + RNN deepfake detection model.

    The model accepts a sequence of video frames
    (batch, sequence_length, height, width, channels) and outputs a
    per-video classification (REAL / FAKE).
    """

    def __init__(self, sequence_length=25, frame_size=224, num_classes=2,
                 backbone="EfficientNetB0", cnn_trainable=False,
                 rnn_units=128, rnn_type="lstm", dropout=0.5):
        """
        Args:
            sequence_length: Number of frames per video sequence.
            frame_size: Spatial size (height = width) of each frame.
            num_classes: Number of output classes.
            backbone: CNN backbone name.
            cnn_trainable: Whether to fine-tune the CNN backbone.
            rnn_units: Number of RNN units.
            rnn_type: 'lstm' or 'gru'.
            dropout: Dropout rate.
        """
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.num_classes = num_classes
        self.cnn_trainable = cnn_trainable
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.backbone = backbone

        self.model = self._build_hybrid()

    def _build_hybrid(self):
        """Build the end-to-end hybrid CNN + RNN model."""
        input_shape = (self.frame_size, self.frame_size, 3)

        # CNN spatial feature extractor (TimeDistributed over frames)
        cnn = CNNFeatureExtractor(
            input_shape=input_shape,
            backbone=self.backbone,
            trainable=self.cnn_trainable,
        )

        # Input: (batch, time, height, width, channels)
        inputs = layers.Input(
            shape=(self.sequence_length,) + input_shape,
            name="video_sequence",
        )

        # Apply CNN to each frame independently
        time_distributed = layers.TimeDistributed(cnn.model)(inputs)

        # RNN temporal model
        rnn = RNNSequenceModel(
            sequence_length=self.sequence_length,
            feature_dim=cnn.get_feature_dim(),
            num_classes=self.num_classes,
            rnn_units=self.rnn_units,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
        )

        # Build RNN on the time-distributed features
        outputs = rnn.model(time_distributed)

        return models.Model(inputs=inputs, outputs=outputs, name="hybrid_deepfake_model")

    def compile(self, learning_rate=1e-4, metrics=None):
        """Compile the model with optimizer and loss."""
        if metrics is None:
            metrics = ["accuracy"]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

    def summary(self):
        """Print a model summary."""
        return self.model.summary()

    def save(self, path):
        """Save the hybrid model to disk."""
        self.model.save(path)

    @classmethod
    def load(cls, path):
        """Load a previously saved hybrid model."""
        loaded = models.load_model(path)
        instance = cls.__new__(cls)
        instance.model = loaded
        return instance
