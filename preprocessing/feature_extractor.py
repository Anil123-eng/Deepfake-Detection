"""
Face detection and region-of-interest extraction.

For deepfake detection, focusing on facial regions is often crucial since
most manipulation targets faces. This module provides face detection and
cropping utilities used before feeding frames to the CNN.
"""

import logging

import cv2

from config import Config
from preprocessing.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class FaceExtractor:
    """
    Detects and crops faces from video frames using Haar cascades.
    """

    def __init__(self, cascade_path=None):
        cascade_path = cascade_path or Config.FACE_DETECTOR_PATH
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            logger.warning(
                "Haar cascade not found at %s. Face cropping disabled.",
                cascade_path,
            )
            self.cascade = None

    def detect_faces(self, frame):
        """
        Detect faces in a single BGR/RGB frame.

        Args:
            frame: Image array (H, W, 3).

        Returns:
            List of bounding boxes (x, y, w, h).
        """
        if self.cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        return faces.tolist() if len(faces) else []

    def extract_face(self, frame, target_size=None, margin=0.3):
        """
        Crop the largest face from a frame and resize it.

        Args:
            frame: RGB image array.
            target_size: (h, w) output size.
            margin: Fraction of box size to expand around the face.

        Returns:
            Cropped face array, or the original frame if no face is found.
        """
        from preprocessing.video_processor import VideoProcessor

        target_size = target_size or (Config.FRAME_SIZE, Config.FRAME_SIZE)
        faces = self.detect_faces(frame)
        if not faces:
            return cv2.resize(frame, target_size)

        # Pick the largest face (most likely the subject of interest)
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        # Expand bounding box with a margin
        mx, my = int(w * margin), int(h * margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(frame.shape[1], x + w + mx)
        y1 = min(frame.shape[0], y + h + my)

        face = frame[y0:y1, x0:x1]
        face = cv2.resize(face, target_size)
        return face


class FeaturePipeline:
    """
    End-to-end feature extraction pipeline for video files:

        video -> sample frames -> face-crop -> normalize -> sequence
    """

    def __init__(self, config=None):
        self.config = config or Config
        self.video_processor = VideoProcessor(self.config)
        self.face_extractor = FaceExtractor()

    def process_video(self, video_path, crop_face=True, max_frames=None):
        """
        Produce a model-ready sequence tensor from a video file.

        Returns:
            np.ndarray of shape (max_frames, frame_size, frame_size, 3).
        """
        max_frames = max_frames or self.config.MAX_FRAMES
        frames = self.video_processor.sample_frames(
            video_path,
            max_frames=max_frames,
            target_size=(self.config.FRAME_SIZE, self.config.FRAME_SIZE),
        )
        if crop_face:
            frames = self.crop_faces_from_sequence(frames)
        return frames

    def crop_faces_from_sequence(self, frames):
        """Apply face cropping to a sequence of normalized frames."""
        # frames are float [0,1]; convert to uint8 for OpenCV
        uint_frames = (frames * 255.0).astype("uint8")
        cropped = []
        for frame in uint_frames:
            face = self.face_extractor.extract_face(frame)
            cropped.append(face)
        cropped = self.video_processor.normalize_frames(cropped)
        return cropped
