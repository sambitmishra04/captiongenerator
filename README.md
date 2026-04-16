# Image Caption Generator

This repo is a Streamlit app plus training notebook for generating captions from
images. The captioning pipeline has two parts:

- A CNN backbone that extracts visual features from the image
- A caption model that predicts words one token at a time

## Current State

The repo contains artifacts from two related setups:

- `VGG16` feature extraction with a `4096`-dim visual vector
- `MobileNetV2` feature extraction with a `1280`-dim visual vector

The training notebook currently builds a VGG16-based model. The runtime code in
`cnn_lstm_captioner.py` now inspects `mymodel.h5` and automatically loads the
matching backbone so inference stays aligned with the saved model.

## Repo Layout

```text
app.py                    Streamlit UI
cnn_lstm_captioner.py     Inference wrapper with backbone auto-detection
image-captioner.ipynb     Training notebook
mymodel.h5                Saved caption model
tokenizer.pkl             Saved tokenizer
flickr8k/                 Local dataset folder
resource/demo.gif         Demo asset
```

## Dataset Layout

The notebook expects Flickr8k captions plus images. It can work with either of
these image layouts:

```text
flickr8k/
  Images/
    *.jpg
  captions.txt
```

or:

```text
flickr8k/
  Flicker8k_Dataset/
    *.jpg
  captions.txt
```

If `captions.txt` is missing but `Flickr8k.token.txt` exists, the notebook will
convert it into `captions.txt`.

## Run The App

1. Install Python 3.10 or 3.11.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

If the app reports a model loading failure, retrain the model from the notebook
in your current TensorFlow environment.

## Retrain The Model

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open `image-captioner.ipynb`.
3. Run all cells.
4. Confirm the notebook saves:
   - `mymodel.h5`
   - `tokenizer.pkl`
5. Start the app again with `streamlit run app.py`.

## Run in Google Colab (GitHub Workflow)

The most efficient way to run this project is to use GitHub for your code and Google Drive for your large model files (`.h5`).

### 1. Push to GitHub
Ensure you have created a repository on GitHub, then run:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git branch -M main
git push -u origin main
```
*Note: Your 2GB dataset and `.h5` files are ignored by `.gitignore` to keep the push fast.*

### 2. Prepare Google Drive
Upload `mymodel.h5` and `tokenizer.pkl` to a folder in your Google Drive (e.g., `ImageCaptioner/`).

### 3. Setup Colab
Open a new notebook in Colab and run:

```python
# Clone your repo
!git clone https://github.com/your-username/your-repo.git
%cd your-repo

# Mount Drive to get your model
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/ImageCaptioner/mymodel.h5 .

# Install dependencies
!pip install -r requirements.txt
```

### 4. Run Inference or Retrain
- **To Generate Captions (UI)**:
  ```bash
  !pip install localtunnel
  !streamlit run app.py & npx localtunnel --port 8501
  ```
- **To Retrain**: Open `image-captioner.ipynb` in Colab and run all cells. The notebook will automatically download the 2GB Flickr8k dataset in seconds using its built-in mirrors.

## Important Compatibility Rule

Training and inference must use the same visual feature size:

- VGG16-trained caption model -> expects `4096`-dim features
- MobileNetV2-trained caption model -> expects `1280`-dim features

The runtime now handles this automatically for supported models, but the saved
model still has to be internally consistent with how it was trained.

## Known Limits

- The checked-in `mymodel.h5` may still be incompatible with some TensorFlow
  versions.
- The notebook is still the source of truth for training.
- This repo does not include automated tests for end-to-end caption quality.
