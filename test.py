"""
==============================================================
🧫 BacNet - Model Test Script
Tests EfficientNet-B0 and ResNet50 (.pth) models on one image
==============================================================

Usage:
    python test.py --image sample.jpg
==============================================================
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0
from PIL import Image
import json
import argparse
from pathlib import Path

# ------------------------------------------------------------
# 🔧 CONFIGURATION
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent / "models"

RESNET_WEIGHTS = BASE_DIR / "resnet50_weights.pth"
EFFICIENTNET_WEIGHTS = BASE_DIR / "efficientnet_b0_weights.pth"

RESNET_LABELS = BASE_DIR / "resnet50_weights.json"
EFFICIENTNET_LABELS = BASE_DIR / "efficientnet_b0_weights.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# 🧠 LOAD MODELS
# ------------------------------------------------------------
def load_model(model_name: str, weights_path: Path, num_classes: int):
    """Load a torchvision model architecture and apply trained weights."""
    if model_name.lower().startswith("resnet"):
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower().startswith("efficientnet"):
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name.")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ------------------------------------------------------------
# 🧩 LOAD LABELS
# ------------------------------------------------------------
def load_labels(label_path: Path):
    with open(label_path, "r") as f:
        return json.load(f)["class_names"]

# ------------------------------------------------------------
# 🧬 PREDICTION PIPELINE
# ------------------------------------------------------------
def predict(model, image_path, labels):
    """Run prediction on one image."""
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    top_idx = probs.argmax()
    top_label = labels[top_idx]
    confidence = probs[top_idx] * 100
    prob_dict = {labels[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

    return top_label, confidence, prob_dict

# ------------------------------------------------------------
# 🧪 MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BacNet models on a single image")
    parser.add_argument("--image", required=True, help="Path to input bacterial image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load labels
    labels_res = load_labels(RESNET_LABELS)
    labels_eff = load_labels(EFFICIENTNET_LABELS)

    # Load models
    print("🚀 Loading models...")
    model_res = load_model("resnet50", RESNET_WEIGHTS, len(labels_res))
    model_eff = load_model("efficientnet_b0", EFFICIENTNET_WEIGHTS, len(labels_eff))

    print("✅ Models loaded successfully.")
    print(f"Using device: {DEVICE}\n")

    # Predict with both models
    print(f"🔍 Testing on image: {image_path.name}\n")

    label_res, conf_res, probs_res = predict(model_res, image_path, labels_res)
    label_eff, conf_eff, probs_eff = predict(model_eff, image_path, labels_eff)

    print("📊 Results:")
    print(f"  🧠 ResNet50 Prediction       → {label_res} ({conf_res:.2f}% confidence)")
    print(f"  ⚡ EfficientNet-B0 Prediction → {label_eff} ({conf_eff:.2f}% confidence)")

    # Optional: Soft voting (ensemble)
    print("\n🤝 Soft Voting (average probabilities):")
    avg_probs = {lbl: (probs_res[lbl] + probs_eff[lbl]) / 2 for lbl in labels_res}
    best_label = max(avg_probs, key=avg_probs.get)
    best_conf = avg_probs[best_label]
    print(f"  🧬 Ensemble Prediction → {best_label} ({best_conf:.2f}% confidence)")
