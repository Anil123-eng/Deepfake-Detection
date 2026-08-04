"""
CNN model module - Spatial Feature Extractor.

This module defines an Object-Oriented CNN architecture responsible for
extracting spatial features from individual video frames. It uses a
transfer-learning backbone (EfficientNetB0) to leverage pre-trained weights
from ImageNet.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class CNNFeatureExtractor:
    """
    CNN-based spatial feature extractor.

    Uses a pre-trained convolutional backbone to produce a compact feature
    vector for each input frame. The backbone can be frozen or fine-tuned.
    """

    BACKBONES = {
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
        "VGG16": tf.keras.applications.VGG16,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
    }

    def __init__(self, input_shape=(224, 224, 3), backbone="EfficientNetB0",
                 trainable=False, pooling="avg"):
        """
        Args:
            input_shape: Spatial dimensions of input frames.
            backbone: Name of the pre-trained CNN backbone.
            trainable: Whether to unfreeze the backbone for fine-tuning.
            pooling: Global pooling strategy after backbone.
        """
        self.input_shape = input_shape
        self.backbone_name = backbone
        self.trainable = trainable
        self.pooling = pooling
        self.model = self._build()

    def _build(self):
        """Construct the base CNN model (without temporal layer)."""
        if self.backbone_name not in self.BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. "
                f"Choose from {list(self.BACKBONES.keys())}"
            )

        base_model = self.BACKBONES[self.backbone_name](
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
            pooling=self.pooling,
        )
        base_model.trainable = self.trainable

        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=self.trainable)
        # Add a small dense layer to compress the feature vector
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        features = layers.Dropout(0.3)(x)

        return models.Model(inputs=inputs, outputs=features, name="cnn_feature_extractor")

    def get_feature_dim(self):
        """Return the output feature dimension of the CNN."""
        return self.model.output_shape[-1]

    def summary(self):
        """Print a model summary for debugging."""
        return self.model.summary()
