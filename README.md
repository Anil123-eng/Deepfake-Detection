# 🛡️ DeepGuard — Deepfake Detection with Hybrid CNN + RNN

A local deep learning system for detecting manipulated video content using a **Hybrid CNN + RNN architecture**, built with **TensorFlow / Keras** and served through a **Flask web application**.

## ✨ Features

- **Hybrid Architecture**: CNN (EfficientNetB0) extracts spatial features per frame; LSTM layers analyze temporal patterns across the sequence.
- **Automated Pipeline**: Frame sampling → face cropping → normalization → sequence generation.
- **Data Augmentation**: Spatial (flip, rotation, brightness, noise) and temporal (frame-drop, patch-shuffle) augmentation for robustness.
- **OOP Design**: Modular classes for models, preprocessing, augmentation, dataset handling, and training.
- **Video & Image Detection**: Detect deepfakes in both videos and still images.
- **Web Interface**: Drag-and-drop video/image upload with animated confidence visualization.
- **REST API**: Programmatic access to prediction via `/api/predict` (video) and `/api/predict_image` (image).
- **Localhost Ready**: Runs on `127.0.0.1` for local development.

---

## 📁 Project Structure

```
Deepfake_Detection/
├── app.py                      # Flask application (predictions, uploads)
├── run.py                      # Application entry point
├── train.py                    # Training CLI script
├── config.py                   # Central configuration
├── requirements.txt
├── README.md
├── data/                       # Dataset folder (real/ & fake/ videos)
│   ├── real/
│   └── fake/
├── models/
│   ├── __init__.py
│   ├── cnn_model.py            # CNN spatial feature extractor
│   ├── rnn_model.py            # RNN temporal model (LSTM/GRU)
│   ├── hybrid_model.py         # Combined CNN + RNN architecture
│   └── model_trainer.py        # Training/validation/save logic
├── preprocessing/
│   ├── __init__.py
│   ├── video_processor.py      # Frame extraction & normalization
│   ├── feature_extractor.py    # Face detection & cropping
│   ├── image_processor.py      # Single-image loading & preprocessing
│   └── data_augmentation.py    # Spatial/temporal augmentation
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   └── index.html
└── trained/                    # Saved model weights
```

---

## 🚀 Quick Start

### 1. Setup Environment

Use Python **3.9–3.11** for the pinned TensorFlow 2.15 dependencies. Python 3.14 is not compatible with this requirements file.

```bash
# Create & activate a Python virtual environment
py -3.11 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **TensorFlow version note**: `requirements.txt` pins TensorFlow 2.15.0. If you're on **Python 3.12+**, you may need to install the latest TensorFlow version instead:
> ```bash
> pip install tensorflow keras opencv-python numpy Pillow flask flask-cors scikit-learn tqdm
> ```

### 3. Prepare Training Data

Organize videos into labeled folders:

```
data/
├── real/
│   ├── authentic_1.mp4
│   ├── authentic_2.mp4
│   └── ...
└── fake/
    ├── deepfake_1.mp4
    ├── deepfake_2.mp4
    └── ...
```

> For best results, use **at least 500 videos per class** (the model was developed with 5000+).

### 4. Train the Model

```bash
python train.py --data data --epochs 20 --batch-size 8
```

The model will be saved to `trained/deepfake_hybrid_model.h5`.

### 5. Run the Web App

```bash
python run.py
```

Open **http://localhost:5000** in your browser. The development server binds only to your local machine by default.

To use another local port or explicitly select development mode:

```bash
# Windows PowerShell
$env:APP_ENV = "development"
$env:PORT = "5000"
python run.py
```

### Local Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Local bind address |
| `PORT` | `5000` | Local HTTP port |
| `APP_ENV` | `development` | Application mode |

---

## 🧠 Model Architecture

```
Input Video (25 frames @ 224×224)
        │
        ▼
┌──────────────────────────────┐
│  TimeDistributed CNN         │
│  EfficientNetB0 (ImageNet)   │
│  → Dense 256 → Dropout       │
└──────────────────────────────┘
        │  per-frame feature vectors
        ▼
┌──────────────────────────────┐
│  LSTM (128 units, return_seq)│
│  LSTM (64 units, dropout)    │
│  → Dense 64 → Dropout        │
└──────────────────────────────┘
        │
        ▼
   Softmax (REAL / FAKE)
```

- **Input**: 25 frames sampled evenly from the video.
- **Face cropping** focuses the CNN on facial regions (where manipulation is most visible).
- **Data augmentation** improves generalization to real-world capture variations.

---

## 📡 REST API

### `POST /api/predict`
Upload a video and receive a prediction.

**Request** (multipart/form-data):
- `video`: the video file (MP4, AVI, MOV, MKV, WEBM, M4V; ≤ 200 MB)

**Response**:
```json
{
  "filename": "sample.mp4",
  "prediction": "FAKE",
  "confidence": 92.31,
  "real_probability": 7.69,
  "fake_probability": 92.31,
  "processed_frames": 25
}
```

### `POST /api/predict_image`
Upload an image and receive a real/fake prediction.

**Request** (multipart/form-data):
- `image`: the image file (JPG, PNG, BMP, WEBP, TIFF)

**Response**:
```json
{
  "filename": "sample.jpg",
  "prediction": "FAKE",
  "confidence": 91.05,
  "real_probability": 8.95,
  "fake_probability": 91.05,
  "media_type": "image"
}
```

### `GET /api/health`
Health check for monitoring.

---

## 🧪 Testing Without a Trained Model

If you haven't trained a model yet, the app will still run — the UI will display a **"Model Not Trained"** badge and prediction requests will return a helpful `503` error until you run:

```bash
python train.py
```

---

## 🔬 Improving Accuracy

- **Increase dataset size** — 5000+ samples per class yields stronger generalization.
- **Fine-tune the CNN backbone** — set `CNN_TRAINABLE = True` in `config.py` for deeper adaptation.
- **Tune hyperparameters** — `RNN_UNITS`, `RNN_DROPOUT`, `EPOCHS`, `LEARNING_RATE`.
- **Try GRU** — switch the RNN cell from LSTM to GRU in `hybrid_model.py`.
- **Experiment with backbones** — `ResNet50`, `VGG16`, `MobileNetV2`.

---

## 📄 License

This project is for educational and research purposes. Users are responsible for compliance with applicable laws regarding synthetic media detection.

