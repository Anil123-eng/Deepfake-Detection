"""
Deepfake Detection Web Application.

A Flask-based web interface that allows users to upload a video or image and
get a real/fake prediction. The app runs on localhost by default.

Usage:
    python run.py
    # or
    python app.py
    # Set APP_ENV=production only when a non-development configuration is needed
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

# Configure TensorFlow for local CPU environments:
#   - Use CPU only for predictable local execution.
#   - Limit memory growth so we don't get killed by the OS.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from config import get_config
from preprocessing.feature_extractor import FeaturePipeline
from preprocessing.image_processor import ImageProcessor

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

# Global models (loaded lazily on first request to keep startup fast,
# especially important on constrained cloud tiers).
_video_model = None
_image_model = None
_model_error = None
_model_loaded = False
_pipeline = FeaturePipeline(config)
_image_processor = ImageProcessor(config)


def get_model():
    """Return the loaded video (hybrid) model, or None if unavailable."""
    return get_video_model()


def get_video_model():
    """Lazy-load the hybrid video model."""
    global _video_model, _model_loaded, _model_error
    if not _model_loaded:
        _model_loaded = True
        if config.MODEL_PATH.exists():
            try:
                logger.info("Loading video model...")
                # Lazy import so TensorFlow is not loaded at module import
                # time and keeps startup lightweight.
                from models.model_trainer import HybridModelTrainer
                _video_model = HybridModelTrainer.load(config.MODEL_PATH)
                logger.info("Video model loaded.")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load video model")
                _model_error = str(exc)
                _video_model = None
        else:
            _video_model = None
    return _video_model


def get_image_model():
    """
    Lazy-load the lightweight image classifier.

    Prefers the dedicated image classifier model. If it is not available,
    a trained video model is loaded anyway since its CNN backbone can also
    classify a single image (as a repeated-frame sequence).
    """
    global _image_model
    if _image_model is not None:
        return _image_model

    img_path = config.TRAINED_DIR / "image_classifier.h5"
    if img_path.exists():
        try:
            logger.info("Loading image classifier...")
            # Lazy import so TensorFlow is not loaded at module import time.
            from models.image_classifier import ImageClassifier
            _image_model = ImageClassifier.load(img_path)
            logger.info("Image classifier loaded.")
            return _image_model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load image classifier: %s", exc)
            _image_model = False  # cache failure -> fall back to video model

    # Fallback to the video model
    logger.warning("No dedicated image classifier found; using video model.")
    return get_video_model()


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for any unhandled error so the client never gets an
    empty / non-JSON response body."""
    logger.exception("Unhandled exception")
    message = str(e) or e.__class__.__name__
    return jsonify({"error": message}), 500


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
    model_ready = (
        config.MODEL_PATH.exists()
        or (config.TRAINED_DIR / "image_classifier.h5").exists()
    )
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
    model = get_image_model()
    if model is None or model is False:
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
        # Determine whether `model` is a dedicated image classifier (expects
        # a 4D image batch) or the fallback video model (expects a 5D
        # sequence). Use the image processor accordingly.
        #
        # The wrapped Keras model may be accessed via `.model` (both
        # ImageClassifier and HybridModelTrainer expose `.model`). Inspect
        # the actual Keras input tensor to decide the correct input shape.
        keras_model = getattr(model, "model", model)
        input_ndim = len(keras_model.inputs[0].shape)
        if input_ndim == 4:
            # Dedicated image classifier: expects (batch, H, W, C).
            img = _image_processor.load_image(upload_path, crop_face=True)
            probs = model.predict(img[np.newaxis, ...])[0]
            # The deployed image-classifier artifact was trained with its
            # output neurons in the order [FAKE, REAL], unlike the hybrid
            # video model's [REAL, FAKE] order. Keep this conversion local to
            # image predictions so video predictions retain their mapping.
            fake_prob = float(probs[0])
            real_prob = float(probs[1])
        else:
            # Fallback: repeat the image into a video-style sequence.
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
