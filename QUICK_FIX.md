# Quick Fix

## If The App Does Not Start

The most common failure is that `mymodel.h5` was saved with an incompatible
TensorFlow or Keras version.

## Fast Recovery

1. Open `image-captioner.ipynb`.
2. Run all cells in the same Python environment you will use for the app.
3. Let the notebook save a fresh `mymodel.h5` and `tokenizer.pkl`.
4. Run:

```bash
streamlit run app.py
```

## What Changed In The Repo

- Inference no longer hard-codes MobileNetV2.
- `cnn_lstm_captioner.py` now inspects the caption model input shape and picks:
  - `VGG16` for `4096`-dim visual features
  - `MobileNetV2` for `1280`-dim visual features
- The app now shows the detected backbone instead of claiming one fixed setup.

## Dataset Reminder

The notebook needs:

- Flickr8k images
- `captions.txt` or `Flickr8k.token.txt`

If captions are missing, retraining will still fail even if the app code is
correct.
