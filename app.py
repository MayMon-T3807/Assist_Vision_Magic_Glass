import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

MODEL = load_model('assistvision_model.keras')

with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = (224, 224)

print('Model loaded!')
print('Classes:', CLASS_NAMES)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    
    img = load_img(img_path, target_size=IMG_SIZE)

    
    img_array = img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)


    predictions = MODEL.predict(img_array, verbose=0)[0]

    top3_indices = np.argsort(predictions)[::-1][:3]

    results = []
    for idx in top3_indices:
        results.append({
            'class':      CLASS_NAMES[idx],
            'label':      CLASS_NAMES[idx].replace('_', ' ').title(),
            'confidence': round(float(predictions[idx]) * 100, 1)
        })

    return results


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
    app.run(debug=True)
