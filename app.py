import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

MODEL = None
CLASS_NAMES = None

def get_model():
    global MODEL, CLASS_NAMES
    if MODEL is None:
        print('Loading model...')
        from tensorflow.keras.models import load_model
        MODEL = load_model('assistvision_model.keras')
        print('Model loaded!')
    if CLASS_NAMES is None:
        with open('class_names.json', 'r') as f:
            CLASS_NAMES = json.load(f)
        print('Class names loaded:', CLASS_NAMES)
    return MODEL, CLASS_NAMES

IMG_SIZE = (224, 224)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    model, class_names = get_model()

    # 1. Load and resize to 224x224
    img = load_img(img_path, target_size=IMG_SIZE)

    # 2. Convert to numpy array
    img_array = img_to_array(img)

    # 3. Normalise 0-255 → 0.0-1.0
    img_array = img_array / 255.0

    # 4. Add batch dimension (224,224,3) → (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Run the model
    predictions = model.predict(img_array, verbose=0)[0]

    # 6. Get top 3 results
    top3_indices = np.argsort(predictions)[::-1][:3]

    results = []
    for idx in top3_indices:
        results.append({
            'class':      class_names[idx],
            'label':      class_names[idx].replace('_', ' ').title(),
            'confidence': round(float(predictions[idx]) * 100, 1)
        })

    return results


# ── Health check route — Azure pings this on startup ──────────────────────
# Returns immediately without loading the model
# This prevents the 504 timeout during startup
@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use JPG, PNG or WEBP'}), 400

    # Make sure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    results = predict_image(save_path)

    return jsonify({
        'success': True,
        'img_url': f'/static/uploads/{filename}',
        'results': results
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
