import pathlib
from fastai.vision.all import *
from flask import Flask, request, render_template, jsonify
import torch

# === Cross-platform fix ===
# Ensures models exported on Windows can be loaded on Linux (Render)
if not hasattr(pathlib, "WindowsPath"):
    pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)

# === Load FastAI models ===
print("🚀 Loading FastAI models...")

# Use your Linux-exported models here
learn_eff = load_learner("models/bacteria_classifier_efficientnet_b0_linux.pkl", cpu=True)
learn_res = load_learner("models/bacteria_classifier_resnet50_linux.pkl", cpu=True)

print("✅ Models loaded successfully.")

# === Global prediction history ===
prediction_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    model_choice = request.form.get("model", "Soft Voting (Ensemble)")
    img = PILImage.create(file.stream)

    # --- Predict using selected model ---
    if model_choice == "EfficientNet-B0":
        label, idx, probs = learn_eff.predict(img)
    elif model_choice == "ResNet50":
        label, idx, probs = learn_res.predict(img)
    elif model_choice == "Soft Voting (Ensemble)":
        _, _, probs_eff = learn_eff.predict(img)
        _, _, probs_res = learn_res.predict(img)
        avg_probs = (probs_eff + probs_res) / 2
        idx = avg_probs.argmax()
        label = learn_eff.dls.vocab[idx]
        probs = avg_probs
    else:
        return jsonify({"error": "Invalid model name"}), 400

    conf = float(probs[idx] * 100)
    probs_dict = {learn_eff.dls.vocab[i]: float(p * 100) for i, p in enumerate(probs)}

    prediction_history.append({
        "model": model_choice,
        "class": label,
        "confidence": round(conf, 2)
    })

    return jsonify({
        "predicted_class": label,
        "confidence": round(conf, 2),
        "probabilities": probs_dict,
        "history": prediction_history[-20:]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
