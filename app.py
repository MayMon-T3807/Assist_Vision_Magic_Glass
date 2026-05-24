import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# fallback paths (VERY IMPORTANT for Azure)
MODEL_PATHS = [
    os.path.join(BASE_DIR, "assistvision_model.keras"),
    "/home/site/wwwroot/assistvision_model.keras",
    "/tmp/assistvision_model.keras"
]

CLASS_PATHS = [
    os.path.join(BASE_DIR, "class_names.json"),
    "/home/site/wwwroot/class_names.json",
    "/tmp/class_names.json"
]

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

MODEL = None
CLASS_NAMES = None


# -----------------------------
# LOAD MODEL + CLASSES
# -----------------------------
def get_model():
    global MODEL, CLASS_NAMES

    # load model once
    if MODEL is None:
        model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError(f"Model not found in any path: {MODEL_PATHS}")

        MODEL = load_model(model_path)

    # load class names once
    if CLASS_NAMES is None:
        class_path = next((p for p in CLASS_PATHS if os.path.exists(p)), None)
        if class_path is None:
            raise FileNotFoundError(f"class_names.json not found in: {CLASS_PATHS}")

        with open(class_path, "r") as f:
            CLASS_NAMES = json.load(f)

    return MODEL, CLASS_NAMES


# -----------------------------
# HELPERS
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

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


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(save_path)

    return jsonify({
        "success": True,
        "img_url": f"/static/uploads/{filename}",
        "results": predict_image(save_path)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
