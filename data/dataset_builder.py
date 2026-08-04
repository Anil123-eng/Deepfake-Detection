"""
Dataset builder module.

Scans a directory of labeled videos, extracts frame sequences, applies
augmentation, and prepares numpy arrays suitable for training the hybrid
CNN + RNN model.

Expected data layout:
    data/
        real/
            video1.mp4
            video2.mp4
            ...
        fake/
            video1.mp4
            video2.mp4
            ...
"""

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import Config
from preprocessing.feature_extractor import FeaturePipeline
from preprocessing.data_augmentation import SequenceAugmenter
from preprocessing.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds training/validation/test datasets from raw video folders.
    """

    def __init__(self, config=None):
        self.config = config or Config
        self.pipeline = FeaturePipeline(self.config)
        self.augmenter = SequenceAugmenter()

    def _label(self, category):
        """Convert category string to one-hot label."""
        # category must be 'real' or 'fake'
        cat = category.lower()
        if cat == "fake":
            return np.array([0.0, 1.0])   # FAKE
        return np.array([1.0, 0.0])       # REAL

    def load_video_sequence(self, video_path, crop_face=True, augment=False):
        """Load a single video into a model-ready sequence."""
        seq = self.pipeline.process_video(video_path, crop_face=crop_face)
        if augment:
            seq = self.augmenter.apply(seq)
        return seq

    def build_from_directory(self, data_dir=None, augment=True,
                             shuffle=True, seed=None):
        """
        Scan data_dir/real and data_dir/fake and build a complete dataset.

        Returns:
            X (n_samples, T, H, W, 3), y (n_samples, 2)
        """
        data_dir = Path(data_dir or self.config.DATA_DIR)
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                "Data directory must contain 'real' and 'fake' sub-folders: "
                f"{real_dir} / {fake_dir}"
            )

        real_videos = sorted(
            p for p in real_dir.glob("*.*")
            if VideoProcessor.is_supported(p)
        )
        fake_videos = sorted(
            p for p in fake_dir.glob("*.*")
            if VideoProcessor.is_supported(p)
        )
        logger.info(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

        X, y = [], []
        for vp in tqdm(real_videos, desc="Processing REAL videos"):
            try:
                seq = self.load_video_sequence(vp, augment=augment)
                X.append(seq)
                y.append(self._label("real"))
            except Exception as exc:
                logger.warning(f"Failed to process {vp}: {exc}")

        for vp in tqdm(fake_videos, desc="Processing FAKE videos"):
            try:
                seq = self.load_video_sequence(vp, augment=augment)
                X.append(seq)
                y.append(self._label("fake"))
            except Exception as exc:
                logger.warning(f"Failed to process {vp}: {exc}")

        if not X:
            raise RuntimeError("No valid videos could be processed.")

        X = np.stack(X)
        y = np.array(y)

        if shuffle:
            rng = np.random.default_rng(seed or self.config.SEED)
            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]

        logger.info(f"Dataset built: X={X.shape}, y={y.shape}")
        return X, y

    def split(self, X, y, val_split=None, test_split=None, seed=None):
        """Split dataset into train/val/test arrays."""
        val_split = val_split or self.config.VALIDATION_SPLIT
        test_split = test_split or self.config.TEST_SPLIT
        rng = np.random.default_rng(seed or self.config.SEED)

        n = len(X)
        indices = rng.permutation(n)
        test_n = int(n * test_split)
        val_n = int(n * val_split)

        test_idx = indices[:test_n]
        val_idx = indices[test_n:test_n + val_n]
        train_idx = indices[test_n + val_n:]

        return {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_val": X[val_idx],
            "y_val": y[val_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
        }
