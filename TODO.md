# TODO - Deepfake Detection for Images & Videos (CNN + RNN)

## Plan Confirmed
- Detect BOTH images and videos using CNN + RNN technology.
- Videos: existing hybrid CNN+RNN (25-frame sequence).
- Images: single image treated as 1-frame sequence through the same hybrid CNN+RNN model.
- Localhost access documented.
- Fix dataset builder to skip non-video files.

## Steps
- [x] 1. Add image config to `config.py` (IMAGE_EXTENSIONS, IMAGE_SIZE).
- [x] 2. Create `preprocessing/image_processor.py` (ImageProcessor + ImageSequenceProcessor).
- [x] 3. Update `preprocessing/__init__.py` to export new image processor.
- [x] 4. Add `/api/predict_image` endpoint to `app.py`.
- [x] 5. Update `templates/index.html` with Image/Video toggle.
- [x] 6. Update `static/js/main.js` for image upload + prediction.
- [x] 7. Update `static/css/style.css` for image mode styling.
- [x] 8. Fix `data/dataset_builder.py` to skip non-video files.
- [x] 9. Update `README.md` with image API + localhost setup docs.
- [x] 10. Test the app (load model, run image & video prediction).
- [x] 11. Fix image prediction shape bug in `app.py` (`/api/predict_image`).
      Root cause: the previous code checked `len(model.input_shape) == 4` against
      the wrapper `.input_shape` (a 3-tuple for the image classifier), so it always
      fell into the fallback branch and passed a 5D `(1, 25, 224, 224, 3)` sequence
      into a 4D `(None, 224, 224, 3)` model -> `ValueError`.
      Fix: inspect the actual wrapped Keras model's input tensor (`getattr(model,
      'model', model).inputs[0].shape`) to choose 4D image vs 5D sequence input.
      Verified end-to-end: `/api/predict_image` returns 200 + valid prediction.
- [x] 12. Reduce startup memory (lazy TensorFlow imports in
      `app.py`). App now imports in ~0.6s with `tensorflow` NOT loaded at module
      import; `/api/health` responds instantly.
