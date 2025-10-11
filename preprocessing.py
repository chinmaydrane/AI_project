import cv2
import numpy as np

RESIZE_SHAPE = (224, 224)

def preprocess_image_inference(image_path):
    """
    Reads image from disk, resizes it, and normalizes.
    image_path: str, path to image
    Returns: np.array of shape (224,224,3), dtype float32
    """
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_color is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    image_color = cv2.resize(image_color, RESIZE_SHAPE)
    image_color = image_color / 255.0  # normalize to [0,1]
    return image_color.astype(np.float32)
