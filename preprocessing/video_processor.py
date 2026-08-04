"""
Video processing module.

Handles extraction of frames from video files and prepares them for the
CNN + RNN hybrid model. Includes frame sampling strategies and
normalization.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Object-oriented video processor for extracting and preparing frames.
    """

    def __init__(self, config=None):
        self.config = config or Config

    def sample_frames(self, video_path, max_frames=None, target_size=None):
        """
        Extract a fixed number of evenly-spaced frames from a video.

        Args:
            video_path: Path to the input video file.
            max_frames: Number of frames to sample (default: config).
            target_size: (height, width) for resizing frames.

        Returns:
            np.ndarray of shape (max_frames, h, w, 3) in float32 [0, 1].
        """
        max_frames = max_frames or self.config.MAX_FRAMES
        target_size = target_size or (self.config.FRAME_SIZE, self.config.FRAME_SIZE)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        # Compute indices of evenly spaced frames
        sample_count = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frames = []
        current = 0
        idx_set = set(indices.tolist())

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current in idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
            current += 1

        cap.release()

        # If we have fewer frames than needed, pad by repeating last frame
        while len(frames) < max_frames:
            frames.append(frames[-1] if frames else np.zeros((*target_size, 3),
                                                              dtype=np.uint8))

        frames = np.array(frames, dtype=np.float32) / 255.0
        return frames

    def extract_all_frames(self, video_path, target_size=None):
        """
        Extract all frames from a video (used for visualization/debugging).

        Returns:
            List of RGB frames as np.uint8 arrays.
        """
        target_size = target_size or (self.config.FRAME_SIZE, self.config.FRAME_SIZE)
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def normalize_frames(frames):
        """Normalize frame values to [0, 1]."""
        return np.asarray(frames, dtype=np.float32) / 255.0

    @staticmethod
    def is_supported(video_path):
        """Check if a file extension is supported."""
        ext = Path(video_path).suffix.lower().lstrip(".")
        return ext in Config.ALLOWED_EXTENSIONS
