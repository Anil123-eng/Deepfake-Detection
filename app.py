"""
Deepfake Detection Web Application.

A Flask-based web interface that allows users to upload a video or image and
get a real/fake prediction. The app is configured to run on the public network
(host=0.0.0.0) so it can be accessed from other devices.

Usage:
    python run.py
    # or
    python app.py
    # Set APP_ENV=production for production configuration
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import get_config
from preprocessing.feature_extractor import FeaturePipeline
from preprocessing.image_processor import ImageProcessor
from models.model_trainer import HybridModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

config = get_config()
config.ensure_directories()

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATE_DIR),
    static_folder=str(config.STATIC_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
CORS(app)  # Enable cross-origin requests for public access

# Global model (lazy-loaded on first request)
_model = None
_model_loaded = False
_pipeline = FeaturePipeline(config)
_image_processor = ImageProcessor(config)


def get_model():
    """Lazy-load the trained hybrid model."""
    global _model, _model_loaded
    if not _model_loaded:
        if config.MODEL_PATH.exists():
            logger.info("Loading trained model...")
            _model = HybridModelTrainer.load(config.MODEL_PATH)
        else:
            logger.warning(
                "No trained model found. Some features will be limited."
            )
            _model = None
        _model_loaded = True
    return _model


def allowed_file(filename):
    """Check if a filename has an allowed video extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def allowed_image_file(filename):
    """Check if a filename has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.IMAGE_EXTENSIONS
    )


@app.route("/")
def index():
    """Render the main upload page."""
    model_ready = config.MODEL_PATH.exists()
    return render_template("index.html", model_ready=model_ready)


@app.route("/api/health")
def health():
    """Health check endpoint for monitoring."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": _model_loaded,
            "model_ready": config.MODEL_PATH.exists(),
            "timestamp": time.time(),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """Handle video upload and return a real/fake prediction."""
    model = get_model()
    if model is None:
        return jsonify(
            {
                "error": "Model not trained yet. Please upload training data "
                         "and run `python train.py` first.",
            }
        ), 503

    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": f"Unsupported file type. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}"}
        ), 400

    # Save upload
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    upload_path = config.UPLOAD_DIR / unique_name
    file.save(str(upload_path))

    try:
        # Process video into sequence
        sequence = _pipeline.process_video(upload_path, crop_face=True)
        # Predict
        seq_batch = np.expand_dims(sequence, axis=0)
        probs = model.predict(seq_batch)[0]
        fake_prob = float(probs[1])
        real_prob = float(probs[0])

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = max(fake_prob, real_prob)

        result = {
            "filename": file.filename,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "processed_frames": int(sequence.shape[0]),
            "media_type": "video",
        }
        logger.info(f"Prediction: {result}")
        return jsonify(result)

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500
    finally:
        # Clean up uploaded file
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/predict_image", methods=["POST"])
def predict_image():
    """Handle image upload and return a real/fake prediction."""
    model = get_model()
    if model is None:
        return jsonify(
            {
                "error": "Model not trained yet. Please upload training data "
                         "and run `python train.py` first.",
            }
        ), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not allowed_image_file(file.filename):
        return jsonify(
            {"error": f"Unsupported file type. Allowed: {sorted(config.IMAGE_EXTENSIONS)}"}
        ), 400

    # Save upload
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    upload_path = config.UPLOAD_DIR / unique_name
    file.save(str(upload_path))

    try:
        # Build a 1-sample sequence from the image (repeated frames)
        # so it can be fed through the hybrid CNN + RNN model.
        seq_batch = _image_processor.to_sequence(upload_path, crop_face=True)
        probs = model.predict(seq_batch)[0]
        fake_prob = float(probs[1])
        real_prob = float(probs[0])

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = max(fake_prob, real_prob)

        result = {
            "filename": file.filename,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "media_type": "image",
        }
        logger.info(f"Image prediction: {result}")
        return jsonify(result)

    except Exception as exc:
        logger.exception("Image prediction failed")
        return jsonify({"error": f"Image prediction failed: {str(exc)}"}), 500
    finally:
        # Clean up uploaded file
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 200 MB."}), 413


if __name__ == "__main__":
    logger.info(f"Starting deepfake detection server on {config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
