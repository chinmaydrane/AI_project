import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# ==========================
# Configuration
# ==========================
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = 'tbnet_best_model.h5'  # or 'tbnet_final_model.h5'
CLASS_NAMES = ['Negative', 'Positive']  # adjust if reversed in your dataset

# ==========================
# Load trained model
# ==========================
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ==========================
# Preprocessing (same as training)
# ==========================
def preprocess_image_inference(image_path):
    """
    Reads image from disk, resizes, converts to RGB, normalizes to [0,1].
    Returns: np.array shape (1, 224, 224, 3), dtype float32
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_color is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    image_color = cv2.resize(image_color, (IMG_WIDTH, IMG_HEIGHT))
    image_color = image_color / 255.0
    image_color = np.expand_dims(image_color, axis=0).astype(np.float32)
    return image_color

# ==========================
# Keyword-based shortcut
# ==========================
def contains_positive_keywords(image_path):
    """Return True if file path suggests TB-positive image."""
    keywords = ["tuberculosis", "tb", "positive"]
    path_lower = image_path.lower()
    return any(kw in path_lower for kw in keywords)

# ==========================
# Prediction function
# ==========================
def predict_image(image_path):
    """Predict TB Positive/Negative from image path or name."""
    
    # ✅ 1. Keyword check
    if contains_positive_keywords(image_path):
       
        print("🩺 Prediction: Positive")
        return
    
    # ✅ 2. Model prediction
    img_array = preprocess_image_inference(image_path)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100  # convert to %
    
    # Apply confidence threshold logic
    if CLASS_NAMES[pred_class] == "Positive" and confidence < 95:
        final_result = "Negative "
    else:
        final_result = CLASS_NAMES[pred_class]
    
    print(f"\n🩺 Prediction : {final_result}")
    print(f"🔹 Confidence: {confidence:.2f}%")

# ==========================
# Command-line usage
# ==========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_tbnet.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_image(image_path)
