"""
RNN model module - Temporal Sequence Learner.

This module defines an Object-Oriented RNN architecture (LSTM/GRU) that learns
temporal patterns from the sequence of CNN-extracted frame features. It is the
"temporal" half of the hybrid CNN + RNN deepfake detector.
"""

from tensorflow.keras import layers, models


class RNNSequenceModel:
    """
    RNN-based temporal model operating on a sequence of feature vectors.

    This model consumes the per-frame feature vectors produced by the CNN
    feature extractor and outputs a sequence-level classification.
    """

    def __init__(self, sequence_length, feature_dim, num_classes=2,
                 rnn_units=128, rnn_type="lstm", dropout=0.5):
        """
        Args:
            sequence_length: Number of frames in the temporal sequence.
            feature_dim: Dimension of each frame's feature vector from CNN.
            num_classes: Number of output classes.
            rnn_units: Number of LSTM/GRU units.
            rnn_type: 'lstm' or 'gru'.
            dropout: Dropout rate applied in the RNN.
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout
        self.model = self._build()

    def _build(self):
        """Construct the temporal RNN model."""
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))

        if self.rnn_type == "lstm":
            rnn = layers.LSTM(
                self.rnn_units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=0.2,
            )(inputs)
            rnn = layers.LSTM(self.rnn_units // 2, dropout=self.dropout)(rnn)
        elif self.rnn_type == "gru":
            rnn = layers.GRU(
                self.rnn_units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=0.2,
            )(inputs)
            rnn = layers.GRU(self.rnn_units // 2, dropout=self.dropout)(rnn)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        # Optionally add an attention layer for better temporal focus
        x = layers.Dense(64, activation="relu")(rnn)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return models.Model(inputs=inputs, outputs=outputs, name="rnn_sequence_model")

    def summary(self):
        """Print a model summary for debugging."""
        return self.model.summary()
