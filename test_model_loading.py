import tensorflow as tf
import h5py


def infer_feature_dim(model):
    try:
        feature_dim = model.inputs[0].shape[-1]
        if hasattr(feature_dim, "value"):
            feature_dim = feature_dim.value
        return int(feature_dim) if feature_dim is not None else None
    except Exception:
        return None


print("=" * 60)
print("TESTING MODEL LOADING")
print("=" * 60)

print("\n[1] Standard TensorFlow load_model:")
loaded_model = None
try:
    loaded_model = tf.keras.models.load_model("mymodel.h5")
    print("    SUCCESS")
except Exception as exc:
    print(f"    FAILED: {type(exc).__name__}")
    print(f"    {str(exc)[:160]}")

print("\n[2] Inferred visual feature size:")
if loaded_model is not None:
    feature_dim = infer_feature_dim(loaded_model)
    if feature_dim == 4096:
        print("    4096 -> VGG16-compatible caption model")
    elif feature_dim == 1280:
        print("    1280 -> MobileNetV2-compatible caption model")
    else:
        print(f"    Unrecognized feature size: {feature_dim}")
else:
    print("    Skipped because the model did not load")

print("\n[3] Check H5 file structure:")
try:
    with h5py.File("mymodel.h5", "r") as file:
        print(f"    Root keys: {list(file.keys())}")
        print(f"    Root attributes: {list(file.attrs.keys())}")
except Exception as exc:
    print(f"    FAILED: {exc}")

print("\n[4] Conclusion:")
if loaded_model is None:
    print("    The saved model could not be loaded in this environment.")
    print("    Retrain from image-captioner.ipynb using the current TensorFlow version.")
else:
    print("    The model loaded successfully.")
    print("    Keep inference aligned with the reported visual feature size.")
