"""
Data augmentation module.

Implements augmentation strategies to improve model robustness and reduce
over-fitting. Augmentations are applied spatially to individual frames and
temporally to frame sequences.
"""

import random

import cv2
import numpy as np


class FrameAugmenter:
    """
    Spatial augmentation for individual frames.
    """

    def __init__(self, flip_prob=0.5, rotation_range=10, brightness_range=(0.8, 1.2),
                 noise_prob=0.3):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.noise_prob = noise_prob

    def apply(self, frame):
        """Apply a random combination of augmentations to one frame."""
        img = frame

        if random.random() < self.flip_prob:
            img = cv2.flip(img, 1)

        if self.rotation_range > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = img.shape[:2]
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h))

        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype("uint8")

        if random.random() < self.noise_prob:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype("uint8")

        return img


class SequenceAugmenter:
    """
    Temporal augmentation for frame sequences.

    Also handles sequence-level alterations such as frame dropping and
    shuffling to make the RNN robust to temporal variations.
    """

    def __init__(self, drop_prob=0.2, shuffle_prob=0.1, frame_augmenter=None):
        self.drop_prob = drop_prob
        self.shuffle_prob = shuffle_prob
        self.frame_augmenter = frame_augmenter or FrameAugmenter()

    def apply(self, sequence):
        """
        Augment a sequence of normalized frames (float [0,1]).

        Args:
            sequence: np.ndarray (T, H, W, 3).

        Returns:
            Augmented sequence with the same shape.
        """
        seq = sequence.copy()
        # Convert to uint8 for spatial augmentations
        seq_u8 = (seq * 255.0).astype("uint8")

        # Randomly drop frames (replace with neighbor) to simulate missing data
        if random.random() < self.drop_prob:
            drop_idx = random.randint(0, len(seq_u8) - 1)
            neighbor = random.choice(
                [i for i in range(len(seq_u8)) if i != drop_idx]
            )
            seq_u8[drop_idx] = seq_u8[neighbor]

        # Randomly shuffle a small patch of the sequence
        if random.random() < self.shuffle_prob and len(seq_u8) > 3:
            start = random.randint(0, len(seq_u8) - 3)
            patch = seq_u8[start:start + 3]
            random.shuffle(patch)
            seq_u8[start:start + 3] = patch

        # Spatial augmentation per frame based on a shared seed for consistency
        rng = random.Random(random.randint(0, 100000))
        augmented = []
        for frame in seq_u8:
            augmented.append(self.frame_augmenter.apply(frame))

        return np.asarray(augmented, dtype=np.float32) / 255.0
