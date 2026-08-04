"""
Image processing module.

Handles loading and preparation of single images for deepfake detection.
For a single image, we build a 1-frame sequence that can be fed through the
same hybrid CNN + RNN architecture used for videos (CNN extracts spatial
features, RNN processes the temporal sequence).
"""

import logging

import cv2
import numpy as np

from config import Config
from preprocessing.feature_extractor import FaceExtractor

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Loads and preprocesses a single image file into a model-ready array.
    """

    def __init__(self, config=None):
        self.config = config or Config
        self.face_extractor = FaceExtractor()

    def load_image(self, image_path, target_size=None, crop_face=True):
        """
        Load an image, optionally crop the face, and normalize to [0, 1].

        Args:
            image_path: Path to the image file.
            target_size: (height, width) output size.
            crop_face: Whether to crop the largest face before resizing.

        Returns:
            np.ndarray of shape (H, W, 3) in float32 [0, 1].
        """
        target_size = target_size or (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

        img = cv2.imread(str(image_path))
        if img is None:
            raise IOError(f"Could not read image file: {image_path}")

        # OpenCV loads as BGR; convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if crop_face:
            img = self.face_extractor.extract_face(img, target_size=target_size)
        else:
            img = cv2.resize(img, target_size)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        return img

    def to_sequence(self, image_path, max_frames=None, target_size=None,
                    crop_face=True):
        """
        Convert a single image into a (1, T, H, W, 3) sequence ready for the
        hybrid CNN + RNN model. The single image is repeated ``max_frames``
        times to form a 1-sample sequence of length ``max_frames``.

        Args:
            image_path: Path to the image file.
            max_frames: Sequence length (default: config.SEQUENCE_LENGTH).
            target_size: (height, width) output size.
            crop_face: Whether to crop the face.

        Returns:
            np.ndarray of shape (1, max_frames, H, W, 3).
        """
        max_frames = max_frames or self.config.SEQUENCE_LENGTH
        target_size = target_size or (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

        img = self.load_image(image_path, target_size=target_size, crop_face=crop_face)

        # Build a sequence by repeating the frame
        sequence = np.repeat(img[np.newaxis, ...], max_frames, axis=0)
        return sequence[np.newaxis, ...]

    @staticmethod
    def is_supported(image_path):
        """Check if a file extension is an allowed image type."""
        from pathlib import Path
        ext = Path(image_path).suffix.lower().lstrip(".")
        return ext in Config.IMAGE_EXTENSIONS
