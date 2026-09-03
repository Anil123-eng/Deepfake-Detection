"""
Lightweight Image Classifier for Deepfake Detection.

This is a purpose-built, lightweight CNN classifier designed for single-image
deepfake detection. It is intentionally much smaller and faster than the
hybrid CNN+RNN video model, making it suitable for local CPU environments
where memory and CPU are limited.

It uses MobileNetV2 (a compact, efficient backbone) as the feature extractor
followed by global pooling and a small classifier head.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class ImageClassifier:
    """
    A lightweight CNN classifier for single images.

    Input:  (batch, height, width, 3)
    Output: (batch, num_classes) softmax probabilities.
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=2,
                 backbone="MobileNetV2", trainable=False, dropout=0.3):
        """
        Args:
            input_shape: (height, width, channels) of the input image.
            num_classes: Number of output classes.
            backbone: Lightweight CNN backbone name.
            trainable: Whether to fine-tune the backbone.
            dropout: Dropout rate in the classifier head.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.trainable = trainable
        self.dropout = dropout
        self.model = self._build()

    def _build(self):
        """Construct the lightweight image classifier."""
        backbone = self._get_backbone()

        base_model = backbone(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
            pooling="avg",
        )
        base_model.trainable = self.trainable

        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=self.trainable)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return models.Model(inputs=inputs, outputs=outputs, name="image_classifier")

    def _get_backbone(self):
        """Return the chosen backbone constructor."""
        backbones = {
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "EfficientNetB0": tf.keras.applications.EfficientNetB0,
            "NASNetMobile": tf.keras.applications.NASNetMobile,
        }
        if self.backbone_name not in backbones:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. "
                f"Choose from {list(backbones.keys())}"
            )
        return backbones[self.backbone_name]

    def compile(self, learning_rate=1e-3, metrics=None):
        """Compile the classifier."""
        if metrics is None:
            metrics = ["accuracy"]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

    def predict(self, images, batch_size=1):
        """Predict class probabilities for a batch of images."""
        return self.model.predict(images, batch_size=batch_size, verbose=0)

    def summary(self):
        """Print a model summary."""
        return self.model.summary()

    def save(self, path):
        """Save the model to disk."""
        self.model.save(path)

    @classmethod
    def load(cls, path):
        """Load a saved image classifier."""
        loaded = models.load_model(path)
        instance = cls.__new__(cls)
        instance.model = loaded
        instance.input_shape = tuple(loaded.inputs[0].shape[1:])
        instance.num_classes = loaded.outputs[0].shape[-1]
        return instance
