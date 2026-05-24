import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

BASE_PATH = "/home/site/wwwroot"

MODEL_PATH = os.path.join(BASE_PATH, "assistvision_model.keras")
CLASS_PATH = os.path.join(BASE_PATH, "class_names.json")

UPLOAD_FOLDER = os.path.join(BASE_PATH, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# =====================================================
# GLOBAL CACHE (IMPORTANT for performance)
# =====================================================
MODEL = None
CLASS_NAMES = None


# =====================================================
# LOAD MODEL + CLASS NAMES (SAFE)
# =====================================================
def get_model():
    global MODEL, CLASS_NAMES

    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        MODEL = load_model(MODEL_PATH)

    if CLASS_NAMES is None:
        if not os.path.exists(CLASS_PATH):
            raise FileNotFoundError(f"Class file not found: {CLASS_PATH}")
        with open(CLASS_PATH, "r") as f:
            CLASS_NAMES = json.load(f)

    return MODEL, CLASS_NAMES


# =====================================================
# HELPERS
# =====================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    model, class_names = get_model()

    img = load_img(img_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]

    return [
        {
            "class": class_names[i],
            "label": class_names[i].replace("_", " ").title(),
            "confidence": round(float(preds[i]) * 100, 1),
        }
        for i in top3
    ]


# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    results = predict_image(save_path)

    return jsonify({
        "success": True,
        "img_url": f"/static/uploads/{filename}",
        "results": results
    })


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
