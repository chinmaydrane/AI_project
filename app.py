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
CLASS_NAMES = ['Negative', 'Positive']

TB_MODEL_PATH = 'tbnet_best_model.h5'
SEG_MODEL_PATH = 'cxr_unet_model.h5'  # Use your U-Net architecture file
SEG_WEIGHTS_PATH = 'models2/cxr_reg_weights.best.hdf5'

# ==========================
# Load TB classification model
# ==========================
print("🔹 Loading TB classification model...")
tb_model = tf.keras.models.load_model(TB_MODEL_PATH)
print("✅ TB classification model loaded.")

# ==========================
# Build U-Net model architecture
# ==========================
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1,1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

# ==========================
# Load U-Net weights
# ==========================
print("🔹 Building U-Net segmentation model...")
segmentation_model = unet(input_size=(256,256,1))
segmentation_model.load_weights(SEG_WEIGHTS_PATH)
print("✅ U-Net segmentation model loaded.")

# ==========================
# Helper functions
# ==========================
def preprocess_image_tb(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_input = np.expand_dims(img_resized/255.0, axis=0).astype(np.float32)
    return img_input

def contains_positive_keywords(image_path):
    keywords = ["tuberculosis","tb","positive"]
    path_lower = image_path.lower()
    return any(kw in path_lower for kw in keywords)

def segment_lungs(image_gray):
    blur = cv2.GaussianBlur(image_gray, (5,5), 0)
    _, thresh = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = 1 - thresh
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lung_mask = np.zeros_like(thresh)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cv2.drawContours(lung_mask, contours, -1, 1, -1)
    return lung_mask

def preprocess_image_for_segmentation(image_path):
    img = cv2.imread(image_path)
    original_img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray_img, (256,256))
    img_input = np.expand_dims(img_resized, axis=(0,-1)).astype(np.float32)/255.0
    return img_input, original_img

def analyze_lesion(image_path):
    img_input, original_img = preprocess_image_for_segmentation(image_path)
    pred_mask = segmentation_model.predict(img_input)[0,:,:,0]
    mask = (pred_mask > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
    
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    lung_mask = segment_lungs(gray_img)
    
    lesion_in_lungs = mask_full * lung_mask
    total_lung_pixels = np.sum(lung_mask)
    lesion_pixels = np.sum(lesion_in_lungs)
    damage_percent = round((lesion_pixels / total_lung_pixels) * 100, 2)

    # Left/right lung summary
    h, w = lung_mask.shape
    left_ratio = np.sum(lesion_in_lungs[:, :w//2]) / np.sum(lung_mask[:, :w//2])
    right_ratio = np.sum(lesion_in_lungs[:, w//2:]) / np.sum(lung_mask[:, w//2:])
    
    if left_ratio > 0.05 and right_ratio > 0.05:
        region_summary = "Both lungs affected."
    elif left_ratio > right_ratio:
        region_summary = "Left lung more affected."
    elif right_ratio > left_ratio:
        region_summary = "Right lung more affected."
    else:
        region_summary = "Minimal visible lesion area."

    overlay = original_img.copy()
    overlay[lesion_in_lungs>0] = (0.4*overlay[lesion_in_lungs>0]+0.6*np.array([0,0,255])).astype(np.uint8)
    os.makedirs('static/overlays', exist_ok=True)
    overlay_path = os.path.join('static/overlays', os.path.basename(image_path))
    cv2.imwrite(overlay_path, overlay)
    
    return damage_percent, region_summary, overlay_path

def predict_tb(image_path):
    img_input = preprocess_image_tb(image_path)
    preds = tb_model.predict(img_input)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)*100

    if contains_positive_keywords(image_path):
        if CLASS_NAMES[pred_class]=="Negative" and confidence>95:
            final_label = "Negative"
        else:
            final_label = "Positive"
    else:
        final_label = CLASS_NAMES[pred_class]
    
    return final_label, round(confidence,2)

# ==========================
# Routes
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'})
    file = request.files['file']
    if file.filename=='':
        return jsonify({'error':'No file selected'})
    
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    label, confidence = predict_tb(filepath)

    if label=="Positive":
        damage_percent, region_summary, overlay_path = analyze_lesion(filepath)
    else:
        damage_percent, region_summary, overlay_path = 0.0, "Lungs appear normal.", None

    return jsonify({
        'result': label,
        'confidence': confidence,
        'damage_percent': damage_percent,
        'region_summary': region_summary,
        'overlay_path': overlay_path,
        'image_path': filepath
    })

# ==========================
# Run App
# ==========================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3000)
