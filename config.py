"""
Configuration module for the Deepfake Detection system.

This module centralizes all configuration parameters using Object-Oriented
Programming principles. The configuration is data-driven and can be easily
extended or overridden based on deployment needs.
"""

import os
from pathlib import Path


class Config:
    """Base configuration class holding all system-wide settings."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    TEMPLATE_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"
    TRAINED_DIR = BASE_DIR / "trained"
    UPLOAD_DIR = BASE_DIR / "uploads"

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    FRAME_SIZE = 224          # Spatial dimensions for CNN input
    MAX_FRAMES = 25           # Number of frames sampled per video
    FPS_SAMPLE = 10           # Frames per second for sampling
    FACE_DETECTOR_PATH = os.getenv(
        "FACE_DETECTOR_PATH",
        str(BASE_DIR / "models" / "haarcascade_frontalface_default.xml"),
    )

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------
    IMAGE_SIZE = 224          # Spatial dimensions for image input
    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    CNN_BACKBONE = "EfficientNetB0"   # Spatial feature extractor
    CNN_TRAINABLE = False             # Freeze backbone during initial training
    RNN_UNITS = 128                   # LSTM units
    RNN_DROPOUT = 0.5                 # Dropout rate in RNN
    NUM_CLASSES = 2                   # REAL / FAKE
    SEQUENCE_LENGTH = 25              # Temporal sequence length

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    SEED = 42
    CLASS_WEIGHTS = [1.0, 1.0]        # Applied if class imbalance exists

    # ------------------------------------------------------------------
    # Web server
    # ------------------------------------------------------------------
    HOST = "0.0.0.0"                  # Public network access
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = False
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB upload limit
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

    # ------------------------------------------------------------------
    # Model artifact
    # ------------------------------------------------------------------
    MODEL_PATH = TRAINED_DIR / "deepfake_hybrid_model.h5"
    LABEL_ENCODER_PATH = TRAINED_DIR / "labels.json"

    @classmethod
    def ensure_directories(cls):
        """Create all required directories if they do not exist."""
        for directory in [
            cls.DATA_DIR,
            cls.TRAINED_DIR,
            cls.UPLOAD_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production/public deployment configuration."""
    DEBUG = False
    # Adjust host/port for public network exposure
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))


def get_config():
    """Factory method returning the appropriate config based on environment."""
    env = os.getenv("APP_ENV", "development").lower()
    if env == "production":
        return ProductionConfig
    return DevelopmentConfig
