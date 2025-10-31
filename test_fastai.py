"""
==============================================================
🧫 BacNet - FastAI Model Test Script (pkl version)
==============================================================

Usage:
    python test_fastai.py --image "path_to_image.jpg"
==============================================================
"""

from fastai.vision.all import *
import argparse
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent / "models"
MODEL_EFF = BASE_DIR / "bacteria_classifier_efficientnet_b0.pkl"
MODEL_RES = BASE_DIR / "bacteria_classifier_resnet50.pkl"

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
def load_models():
    print("🚀 Loading FastAI models...")
    learn_eff = load_learner(MODEL_EFF)
    learn_res = load_learner(MODEL_RES)
    print("✅ Models loaded successfully.")
    print(f"Classes: {learn_eff.dls.vocab}\n")
    return learn_eff, learn_res

# ------------------------------------------------------------
# RUN PREDICTION
# ------------------------------------------------------------
def predict_with_models(image_path: Path, learn_eff, learn_res):
    print(f"🔍 Testing on: {image_path.name}\n")

    # Create FastAI image
    img = PILImage.create(image_path)

    # EfficientNet-B0
    label_eff, idx_eff, probs_eff = learn_eff.predict(img)
    conf_eff = float(probs_eff[idx_eff] * 100)

    # ResNet50
    label_res, idx_res, probs_res = learn_res.predict(img)
    conf_res = float(probs_res[idx_res] * 100)

    # Soft voting ensemble
    avg_probs = (probs_eff + probs_res) / 2
    idx_ens = avg_probs.argmax()
    label_ens = learn_eff.dls.vocab[idx_ens]
    conf_ens = float(avg_probs[idx_ens] * 100)

    print("📊 Results:")
    print(f"  ⚡ EfficientNet-B0 → {label_eff} ({conf_eff:.2f}%)")
    print(f"  🧠 ResNet50        → {label_res} ({conf_res:.2f}%)")
    print(f"  🤝 Ensemble (Avg)  → {label_ens} ({conf_ens:.2f}%)")

# ------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input bacterial image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    learn_eff, learn_res = load_models()
    predict_with_models(image_path, learn_eff, learn_res)
