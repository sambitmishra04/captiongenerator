# Architecture

## Runtime

The app uses:

```text
uploaded image
  -> CNN feature extractor
  -> visual feature vector
  -> caption model
  -> token-by-token caption
```

## Supported Backbones

The repo now supports two inference paths, selected automatically from the saved
caption model input shape:

| Backbone | Feature size | When used |
|---|---:|---|
| VGG16 | 4096 | Caption model input is `(4096,)` |
| MobileNetV2 | 1280 | Caption model input is `(1280,)` |

## Training Notebook

`image-captioner.ipynb` currently defines a VGG16-based encoder and an
attention-style caption decoder.

Key notebook properties:

- visual input shape: `4096`
- text input shape: `max_caption_length`
- sequence model: bidirectional LSTM blocks
- output: vocabulary-sized softmax

## App Layer

`app.py` is only a UI wrapper. It does not define the caption model. Its job is
to:

- load the captioner once
- show readiness errors cleanly
- accept single-image or batch uploads
- display generated captions

## Main Risk That Was Fixed

Before this cleanup, the repo claimed a MobileNetV2 inference path while the
training notebook still built a VGG16-based caption model. That creates a hard
feature mismatch:

- VGG16 encoder output: `4096`
- MobileNetV2 encoder output: `1280`

The runtime now detects the expected feature size and loads the matching CNN
backbone automatically.
