from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from PIL import Image

# --- Compatibility Bridge ---
# If tokenizer.pkl was saved with standalone 'keras', but we only have 'tensorflow.keras',
# we must inject the new paths into sys.modules so pickle.load can find them.
try:
    import keras.preprocessing.text
except ImportError:
    try:
        from tensorflow.keras.preprocessing import text, sequence
        # Create a dummy keras module structure
        class MockModule: pass
        km = MockModule()
        km.preprocessing = MockModule()
        km.preprocessing.text = text
        km.preprocessing.sequence = sequence
        sys.modules['keras'] = km
        sys.modules['keras.preprocessing'] = km.preprocessing
        sys.modules['keras.preprocessing.text'] = text
        sys.modules['keras.preprocessing.sequence'] = sequence
    except ImportError:
        pass
# -----------------------------
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as vgg16_preprocess_input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ForgivingLSTM(tf.keras.layers.LSTM):
    """LSTM layer that ignores legacy Keras 2 arguments during Keras 3 loading."""
    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)  # Remove legacy arg that breaks Keras 3
        return super().from_config(config)


class CNNLSTMCaptioner:
    """Unified interface for CNN + caption-model inference."""

    SUPPORTED_BACKBONES = {
        1280: "mobilenetv2",
        4096: "vgg16",
    }

    def __init__(
        self,
        model_path: str = "mymodel.h5",
        tokenizer_path: str = "tokenizer.pkl",
        max_caption_length: int = 34,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        # Unified discovery logic: Check local, then script dir, then common Drive paths
        script_dir = Path(__file__).parent.resolve()
        
        # We check multiple common names the user might have used for their Drive folder
        drive_paths = [
            Path("/content/drive/MyDrive/ImageCaptionProject"),
            Path("/content/drive/MyDrive/ImageCaptioner"),
            Path("/content/drive/MyDrive/captiongenerator"),
            Path("/content/drive/MyDrive/image-caption-generator-master"),
            Path("/content/drive/MyDrive/"), # as a last resort
        ]
        
        search_dirs = [script_dir] + drive_paths
        
        self.model_path = self._resolve_robust_path(model_path, search_dirs)
        self.tokenizer_path = self._resolve_robust_path(tokenizer_path, search_dirs)
        
        self.max_caption_length = max_caption_length
        self.image_size = image_size
        self.errors: list[str] = []

        self.cnn_model: Model | None = None
        self.lstm_model = None
        self.tokenizer = None
        self.index_word: dict[int, str] = {}
        self.backbone_name: str | None = None
        self.feature_dim: int | None = None
        self._preprocess_input = None

        self._load_artifacts()

    def _resolve_robust_path(self, target: str, search_dirs: list[Path]) -> str:
        """Search for a file in multiple locations, returning the first valid one."""
        target_path = Path(target)
        
        # 1. Handle absolute paths
        if target_path.is_absolute():
            return str(target_path) if target_path.exists() else str(target_path)
            
        # 2. Search through prioritized directories
        for base in search_dirs:
            candidate = (base / target).resolve()
            if candidate.exists():
                return str(candidate)
        
        # 3. Fallback to first search dir (usually script dir) to trigger standard errors later
        return str(search_dirs[0] / target)

    def _load_artifacts(self) -> None:
        """Load caption model, tokenizer, and matching CNN backbone."""
        self._load_lstm_model()
        self._load_tokenizer()
        self._load_cnn_model()

    def _load_lstm_model(self) -> None:
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.lstm_model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={'LSTM': ForgivingLSTM},
                compile=False
            )
            self.feature_dim = self._infer_visual_feature_dim()
        except FileNotFoundError as exc:
            self.errors.append(str(exc))
            self.lstm_model = None
        except Exception as exc:  # pragma: no cover - environment-specific
            full_trace = traceback.format_exc()
            self.errors.append(f"Caption model load failed: {exc}\n\nTraceback:\n{full_trace}")
            self.lstm_model = None

    def _infer_visual_feature_dim(self) -> int | None:
        if self.lstm_model is None:
            return None

        try:
            # Look at the first input (expected visual features)
            visual_input = self.lstm_model.inputs[0]
            
            # The shape might be a list of ints, or a Dimension object in older TF,
            # or a symbolic KerasTensor shape in Keras 3.
            shape = getattr(visual_input, "shape", None)
            if shape is None or len(shape) < 2:
                return None

            # Feature dimension is the last dimension of the first input
            feature_dim = shape[-1]
            
            # Extract raw int if it's a Dimension object or symbolic proxy
            if hasattr(feature_dim, "value"):
                feature_dim = feature_dim.value
            
            # If still None (dynamic shape), we might need to check the model config
            if feature_dim is None:
                config = self.lstm_model.get_config()
                # This is a deep dive, usually only needed for very dynamic models
                try:
                    layers = config.get("layers", [])
                    if layers:
                        first_input_layer = layers[0]
                        batch_input_shape = first_input_layer.get("config", {}).get("batch_input_shape")
                        if batch_input_shape:
                            feature_dim = batch_input_shape[-1]
                except Exception:
                    pass

            return int(feature_dim) if feature_dim is not None else None
        except Exception as exc:  # pragma: no cover - defensive
            self.errors.append(f"Could not infer caption model input shape: {exc}")
            return None

    def _load_tokenizer(self) -> None:
        try:
            with open(self.tokenizer_path, "rb") as file:
                self.tokenizer = pickle.load(file)

            if hasattr(self.tokenizer, "index_word") and self.tokenizer.index_word:
                self.index_word = dict(self.tokenizer.index_word)
            elif hasattr(self.tokenizer, "word_index"):
                self.index_word = {
                    index: word for word, index in self.tokenizer.word_index.items()
                }
        except FileNotFoundError:
            self.errors.append(f"Tokenizer file not found: {self.tokenizer_path}")
            self.tokenizer = None
        except Exception as exc:  # pragma: no cover - environment-specific
            self.errors.append(f"Tokenizer load failed: {exc}")
            self.tokenizer = None

    def _load_cnn_model(self) -> None:
        if self.feature_dim is None:
            if self.lstm_model is not None:
                self.errors.append(
                    "Could not determine which CNN backbone the caption model expects."
                )
            return

        backbone_name = self.SUPPORTED_BACKBONES.get(self.feature_dim)
        if backbone_name is None:
            self.errors.append(
                f"Unsupported caption model input size {self.feature_dim}. "
                "Expected 1280 (MobileNetV2) or 4096 (VGG16)."
            )
            return

        try:
            if backbone_name == "mobilenetv2":
                backbone = MobileNetV2(weights="imagenet")
                self.cnn_model = Model(
                    inputs=backbone.inputs,
                    outputs=backbone.layers[-2].output,
                )
                self._preprocess_input = mobilenet_preprocess_input
            else:
                backbone = VGG16(weights="imagenet")
                self.cnn_model = Model(
                    inputs=backbone.inputs,
                    outputs=backbone.layers[-2].output,
                )
                self._preprocess_input = vgg16_preprocess_input

            self.backbone_name = backbone_name
        except Exception as exc:  # pragma: no cover - environment-specific
            self.errors.append(f"CNN backbone load failed: {exc}")
            self.cnn_model = None

    def is_ready(self) -> bool:
        return (
            self.cnn_model is not None
            and self.lstm_model is not None
            and self.tokenizer is not None
        )

    def get_errors(self) -> list[str]:
        return self.errors

    def get_runtime_info(self) -> dict[str, object]:
        return {
            "ready": self.is_ready(),
            "backbone": self.backbone_name,
            "feature_dim": self.feature_dim,
            "max_caption_length": self.max_caption_length,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "errors": list(self.errors),
        }

    def _extract_image_features(self, image_array: np.ndarray) -> np.ndarray:
        if self.cnn_model is None or self._preprocess_input is None:
            raise ValueError("CNN backbone is not loaded")

        preprocessed = self._preprocess_input(image_array.copy())
        return self.cnn_model.predict(preprocessed, verbose=0)

    def _get_word_from_index(self, index: int) -> str:
        return self.index_word.get(index, "")

    def _generate_caption_lstm(self, image_features: np.ndarray) -> str:
        caption = "startseq"

        for _ in range(self.max_caption_length):
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences(
                [sequence],
                maxlen=self.max_caption_length,
                padding="post",
            )

            predictions = self.lstm_model.predict([image_features, sequence], verbose=0)
            predicted_index = int(np.argmax(predictions))
            predicted_word = self._get_word_from_index(predicted_index)

            if not predicted_word:
                break

            caption += " " + predicted_word
            if predicted_word == "endseq":
                break

        return caption

    def _load_image(self, image_input) -> Image.Image:
        if isinstance(image_input, str):
            return load_img(image_input, target_size=self.image_size)

        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input.astype("uint8")).convert("RGB").resize(
                self.image_size
            )

        return Image.open(image_input).convert("RGB").resize(self.image_size)

    def generate_caption(self, image_input) -> str | None:
        """Extract features and generate a human-readable caption."""
        if not self.is_ready():
            return None

        try:
            image = self._load_image(image_input)
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)

            image_features = self._extract_image_features(image_array)
            raw_caption = self._generate_caption_lstm(image_features)
            
            # Strip tokens and normalize whitespace
            clean_caption = raw_caption.replace("startseq", "").replace("endseq", "")
            return " ".join(clean_caption.split())
        except Exception as exc:  # pragma: no cover - runtime safety
            self.errors.append(f"Caption generation failed: {exc}")
            return None

    def generate_captions_batch(self, image_inputs: Iterable) -> list[dict[str, object]]:
        results = []
        for index, image_input in enumerate(image_inputs):
            caption = self.generate_caption(image_input)
            results.append(
                {
                    "index": index,
                    "input": image_input,
                    "caption": caption,
                    "success": caption is not None,
                }
            )
        return results
