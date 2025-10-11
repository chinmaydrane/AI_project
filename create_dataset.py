import os
import glob
import cv2
import argparse
from preprocessing import preprocess_image

# ------------------------
# Arguments
# ------------------------
parser = argparse.ArgumentParser(description='TB-Net Dataset Creation')
parser.add_argument(
    '--datapath',
    default='Dataset',
    type=str,
    help='The root folder containing the "Normal" and "Tuberculosis" folders.'
)
default_output = os.path.join(os.path.dirname(__file__), "data")
parser.add_argument(
    "--outputpath",
    default=default_output,
    help="Path to save preprocessed dataset"
)
args = parser.parse_args()

# ------------------------
# Make sure output folder exists
# ------------------------
os.makedirs(args.outputpath, exist_ok=True)

# ------------------------
# Gather all image files
# ------------------------
extensions = ("*.png", "*.jpg", "*.jpeg")
filenames = []

for cls in ["Normal", "Tuberculosis"]:
    class_path = os.path.join(args.datapath, cls)
    for ext in extensions:
        filenames.extend(glob.glob(os.path.join(class_path, ext)))

print(f"Found {len(filenames)} images.")

# ------------------------
# Process and save images
# ------------------------
count = 0
for filename in filenames:
    try:
        image = preprocess_image(filename)

        # Determine label and create subfolder
        label = "Normal" if "Normal" in filename else "Tuberculosis"
        class_dir = os.path.join(args.outputpath, label)
        os.makedirs(class_dir, exist_ok=True)

        # Save inside that subfolder
        basename = os.path.basename(filename)
        savepath = os.path.join(class_dir, os.path.splitext(basename)[0] + ".png")

        success = cv2.imwrite(savepath, image)
        if not success:
            print("❌ Failed to save:", savepath)

    except Exception as e:
        print("⚠️ Error processing:", filename, e)

    # Progress printing
    if count % 500 == 0:
        print(f"{len(filenames) - count} images remaining.")
    count += 1

print("✅ Done! All images saved to:", args.outputpath)
