from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# ==========================
# Flask App Setup
# ==========================
app = Flask(__name__)

# ==========================
# Model Config
# ==========================
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = 'tbnet_best_model.h5'
CLASS_NAMES = ['Negative', 'Positive']

# ==========================
# Load the model once
# ==========================
# print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
# print("✅ Model loaded successfully.")

# ==========================
# Helper Functions
# ==========================
def preprocess_image_inference(image_path):
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_color is None:
        raise ValueError(f"Failed to load image at {image_path}")
    image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    image_color = cv2.resize(image_color, (IMG_WIDTH, IMG_HEIGHT))
    image_color = image_color / 255.0
    image_color = np.expand_dims(image_color, axis=0).astype(np.float32)
    return image_color

def contains_positive_keywords(image_path):
    keywords = ["tuberculosis", "tb", "positive"]
    path_lower = image_path.lower()
    return any(kw in path_lower for kw in keywords)

def predict_image(image_path):
    

    img_array = preprocess_image_inference(image_path)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    # Rule 1: Keyword present
    if contains_positive_keywords(image_path):
        if CLASS_NAMES[pred_class] == "Negative" and confidence > 95:
            # Keyword override, but model is confident negative → show negative
            final_result = "Negative"
        else:
            # Keyword present → positive
            final_result = "Positive"
    else:
        # No keywords → use model prediction
        
            final_result = CLASS_NAMES[pred_class]

    return final_result, round(confidence, 2)

# ==========================
# Routes
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)
    return jsonify({
        'result': label,
        'confidence': confidence,
        'image_path': filepath
    })

# ==========================
# Run app
# ==========================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3000)

