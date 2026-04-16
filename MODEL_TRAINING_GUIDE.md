# Model Training Guide

## Goal

Train a caption model and save artifacts that the Streamlit app can load
reliably.

## Prerequisites

- Python 3.10 or 3.11
- Installed dependencies from `requirements.txt`
- Jupyter Notebook
- Flickr8k images and captions available under `flickr8k/`

Install notebook support if needed:

```bash
pip install jupyter notebook
```

## Start Training

```bash
jupyter notebook
```

Open `image-captioner.ipynb` and run all cells.

## Dataset Requirements

The notebook looks for a dataset root under `flickr8k/` and supports either:

```text
flickr8k/Images/*.jpg
```

or:

```text
flickr8k/Flicker8k_Dataset/*.jpg
```

For captions it expects one of:

- `captions.txt`
- `Flickr8k.token.txt`

If only `Flickr8k.token.txt` exists, the notebook can derive `captions.txt`.

## Backbone Compatibility

The original notebook architecture is VGG16-based.

That means:

- the encoder emits `4096`-dim feature vectors
- the caption model input shape is `(4096,)`
- inference must also use VGG16 features unless you retrain the full pipeline
  for another backbone

If you change the notebook to MobileNetV2, you must retrain the caption model
so the saved `mymodel.h5` expects `1280`-dim features.

## Saved Outputs

After training, the notebook should save:

- `mymodel.h5`
- `tokenizer.pkl`

These files must live in the repo root for the default app configuration.

## Run The App After Training

```bash
streamlit run app.py
```

The app will inspect `mymodel.h5` and load the correct supported backbone for
inference.

## Common Failures

| Issue | Meaning | Fix |
|---|---|---|
| `bad marshal data` | Saved model is incompatible with current TensorFlow/Keras | Retrain in the current environment |
| `Images directory was not found` | Notebook is pointed at the wrong dataset layout | Place images under `flickr8k/Images` or `flickr8k/Flicker8k_Dataset` |
| caption file missing | Training data is incomplete | Add `captions.txt` or `Flickr8k.token.txt` |
| shape mismatch at inference | Backbone used for inference differs from training | Retrain or use a model saved for the same feature size |
