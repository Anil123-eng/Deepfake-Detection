"""
Utility script for generating synthetic test videos.

This script creates small synthetic videos so you can verify the
training pipeline and web app without a real dataset.

It creates:
    data/real/  - solid-color videos labeled as REAL
    data/fake/  - noisy / flickering videos labeled as FAKE

Usage:
    python utils/generate_test_data.py --real 10 --fake 10
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


def _write_video(path, frames, fps=10, size=(224, 224)):
    """Write a list of frames as an mp4 video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def generate_real_video(path, frames=25, size=(224, 224), seed=None):
    """Generate a synthetic 'real' video (smooth moving color gradient)."""
    rng = np.random.default_rng(seed)
    frames_list = []
    base = rng.integers(0, 255, size=(size[0] // 2, size[1] // 2, 3), dtype="uint8")
    for i in range(frames):
        # Slowly shift brightness = smooth temporal change
        factor = 1.0 + 0.02 * np.sin(i / 2.0)
        frame = np.clip(base.astype(np.float32) * factor, 0, 255).astype("uint8")
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        frames_list.append(frame)
    _write_video(path, frames_list)


def generate_fake_video(path, frames=25, size=(224, 224), seed=None):
    """Generate a synthetic 'fake' video (high noise + abrupt flicker)."""
    rng = np.random.default_rng(seed)
    frames_list = []
    for i in range(frames):
        noise = rng.integers(0, 255, size=(size[0], size[1], 3), dtype="uint8")
        # Abrupt flicker every few frames simulates artifact
        if i % 5 < 2:
            noise = np.clip(noise.astype(np.int16) + 60, 0, 255).astype("uint8")
        frames_list.append(noise)
    _write_video(path, frames_list)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test videos")
    parser.add_argument("--real", type=int, default=10, help="Number of 'real' videos")
    parser.add_argument("--fake", type=int, default=10, help="Number of 'fake' videos")
    parser.add_argument("--frames", type=int, default=25, help="Frames per video")
    args = parser.parse_args()

    real_dir = Config.DATA_DIR / "real"
    fake_dir = Config.DATA_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.real} synthetic REAL videos...")
    for i in range(args.real):
        generate_real_video(real_dir / f"real_{i:03d}.mp4", frames=args.frames, seed=i)
    print(f"Generating {args.fake} synthetic FAKE videos...")
    for i in range(args.fake):
        generate_fake_video(fake_dir / f"fake_{i:03d}.mp4", frames=args.frames, seed=i)

    print("Done! Synthetic dataset created in data/real and data/fake")


if __name__ == "__main__":
    main()
