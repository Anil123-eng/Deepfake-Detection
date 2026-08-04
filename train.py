"""
Training script for the hybrid CNN + RNN deepfake detection model.

Usage:
    python train.py [--data <path>] [--epochs N] [--batch-size N] [--no-augment]

This script:
    1. Builds the dataset from a real/fake video folder structure.
    2. Splits into train/val/test.
    3. Trains the hybrid CNN + RNN model.
    4. Evaluates and saves the model.
"""

import argparse
import logging

from config import get_config
from preprocessing.feature_extractor import FeaturePipeline
from data.dataset_builder import DatasetBuilder
from models.model_trainer import HybridModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train hybrid deepfake model")
    parser.add_argument("--data", help="Path to data directory (real/ & fake/)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--save", help="Path to save the trained model")
    args = parser.parse_args()

    config = get_config()
    config.ensure_directories()

    # 1. Build dataset
    logger.info("Building dataset...")
    builder = DatasetBuilder(config)
    X, y = builder.build_from_directory(
        data_dir=args.data,
        augment=not args.no_augment,
    )
    if len(X) < 10:
        raise RuntimeError("Not enough videos to train. Add more data.")

    splits = builder.split(X, y)
    logger.info(
        f"Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, "
        f"Test: {len(splits['X_test'])}"
    )

    # 2. Train hybrid model
    logger.info("Initializing hybrid CNN + RNN model...")
    trainer = HybridModelTrainer(config=config)
    trainer.model.summary()

    trainer.train(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # 3. Evaluate
    metrics = trainer.evaluate(splits["X_test"], splits["y_test"])
    logger.info(f"Final test accuracy: {metrics['accuracy']:.4f}")

    # 4. Save model
    trainer.save(args.save)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
