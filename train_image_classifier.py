"""
Training script for the lightweight image classifier.

This builds a lightweight MobileNetV2-based image classifier from the existing
video dataset by sampling frames from the real/ and fake/ videos. It is much
faster and lighter than the full hybrid CNN+RNN model, making it suitable for
single-image detection on constrained cloud deployments (e.g. Render free tier).

Usage:
    python train_image_classifier.py [--data <path>] [--epochs N]
                                     [--backbone MobileNetV2]
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_image_cls")

from config import Config, get_config  # noqa: E402
from models.image_classifier import ImageClassifier  # noqa: E402


def build_frame_dataset(data_dir, max_per_video=5, target_size=(224, 224),
                        seed=42):
    """
    Sample frames from all videos and build an (n_images, H, W, 3) dataset.

    Returns X (np.float32, [0,1]) and y (one-hot, n_classes).
    """
    import cv2
    from preprocessing.feature_extractor import FaceExtractor

    data_dir = Path(data_dir or Config.DATA_DIR)
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Data dir must contain 'real' and 'fake'. Got: {data_dir}"
        )

    face = FaceExtractor()
    X, y = [], []

    def process(category, path, label):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.warning("Could not open %s", path)
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return
        # Evenly-spaced indices to sample
        count = min(max_per_video, total)
        idx = np.linspace(0, total - 1, count, dtype=int)
        cur = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cur in idx:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    img = face.extract_face(frame, target_size=target_size)
                except Exception:
                    img = cv2.resize(frame, target_size)
                X.append(img.astype(np.float32) / 255.0)
                y.append(label)
            cur += 1
        cap.release()

    logger.info("Processing REAL videos...")
    for vp in sorted(real_dir.glob("*.mp4")):
        process("real", vp, [1.0, 0.0])
    logger.info("Processing FAKE videos...")
    for vp in sorted(fake_dir.glob("*.mp4")):
        process("fake", vp, [0.0, 1.0])

    if not X:
        raise RuntimeError("No valid frame data could be built. Add videos first.")

    X = np.stack(X)
    y = np.array(y)
    # Shuffle
    rng = np.random.default_rng(seed)
    ord = rng.permutation(len(X))
    logger.info("Frame dataset: X=%s y=%s", X.shape, y.shape)
    return X[ord], y[ord]


def main():
    parser = argparse.ArgumentParser(description="Train lightweight image classifier")
    parser.add_argument("--data", help="Path to data directory (real/ & fake/)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone", default="MobileNetV2",
                        choices=["MobileNetV2", "EfficientNetB0", "NASNetMobile"])
    parser.add_argument("--save", help="Path to save the model (else config path)")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    config = get_config()
    config.ensure_directories()

    logger.info("Building frame dataset...")
    X, y = build_frame_dataset(args.data, max_per_video=5)

    # Split train/val
    n = len(X)
    val_n = int(n * args.val_split)
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(n)
    X_val, y_val = X[perm[:val_n]], y[perm[:val_n]]
    X_tr, y_tr = X[perm[val_n:]], y[perm[val_n:]]
    logger.info("Train: %d, Val: %d", len(X_tr), len(X_val))

    logger.info("Building lightweight image classifier (%s)...", args.backbone)
    classifier = ImageClassifier(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES,
        backbone=args.backbone,
        trainable=False,
        dropout=0.3,
    )
    classifier.compile(learning_rate=1e-3)
    classifier.summary()

    logger.info("Starting training...")
    history = classifier.model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
    )

    save_path = args.save or str(config.TRAINED_DIR / "image_classifier.h5")
    classifier.save(save_path)
    logger.info("Image classifier saved to %s", save_path)


if __name__ == "__main__":
    main()
