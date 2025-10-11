import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =============================
# Setup paths
# =============================
RAW_PATH = "Dataset"
PROC_PATH = "data"
RESULTS_PATH = "eda_results"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =============================
# Helper Functions
# =============================
def get_image_paths(base_path):
    classes = ["Normal", "Tuberculosis"]
    data = []
    for cls in classes:
        paths = glob(os.path.join(base_path, cls, "*.png"))
        for p in paths:
            data.append((p, cls))
    return pd.DataFrame(data, columns=["path", "label"])

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def save_plot(fig, name):
    fig.savefig(os.path.join(RESULTS_PATH, name), bbox_inches="tight")
    plt.close(fig)

# =============================
# Load Data
# =============================
raw_df = get_image_paths(RAW_PATH)
proc_df = get_image_paths(PROC_PATH)

print("Raw dataset size:", len(raw_df))
print("Processed dataset size:", len(proc_df))

# =============================
# 1. Class Distribution
# =============================
fig, ax = plt.subplots()
raw_df["label"].value_counts().plot(kind="bar", ax=ax, title="Class Distribution (Raw)")
save_plot(fig, "class_distribution_raw.png")

fig, ax = plt.subplots()
proc_df["label"].value_counts().plot(kind="bar", ax=ax, title="Class Distribution (Processed)")
save_plot(fig, "class_distribution_processed.png")

# =============================
# 2. Image Dimensions (Raw)
# =============================
dims = []
for path in raw_df["path"]:
    img = load_image(path)
    dims.append(img.shape)
dims_df = pd.DataFrame(dims, columns=["height", "width"])
fig, ax = plt.subplots()
dims_df.hist(ax=ax)
save_plot(fig, "image_dimensions_raw.png")

# =============================
# 3. Pixel Intensity Distribution
# =============================
for label in raw_df["label"].unique():
    pixels = []
    subset = raw_df[raw_df["label"] == label]["path"].sample(min(200, len(raw_df)))
    for path in subset:
        img = load_image(path)
        pixels.extend(img.flatten())
    fig, ax = plt.subplots()
    ax.hist(pixels, bins=50, color="blue", alpha=0.7)
    ax.set_title(f"Pixel Intensity Distribution ({label})")
    save_plot(fig, f"pixel_intensity_{label}.png")

# =============================
# 4. Sample Images Before vs After
# =============================
samples = raw_df.sample(4, random_state=42)
fig, axes = plt.subplots(4, 2, figsize=(6, 12))
for i, row in enumerate(samples.itertuples()):
    raw = load_image(row.path)
    proc_path = row.path.replace("Dataset", "data")
    if os.path.exists(proc_path):
        proc = load_image(proc_path)
    else:
        proc = np.zeros_like(raw)
    axes[i,0].imshow(raw, cmap="gray"); axes[i,0].set_title(f"Raw - {row.label}")
    axes[i,1].imshow(proc, cmap="gray"); axes[i,1].set_title("Processed")
    axes[i,0].axis("off"); axes[i,1].axis("off")
save_plot(fig, "samples_before_after.png")

# =============================
# 5. Mean Image per Class
# =============================
for label in raw_df["label"].unique():
    imgs = []
    subset = raw_df[raw_df["label"] == label]["path"].sample(min(100, len(raw_df)))
    for path in subset:
        img = cv2.resize(load_image(path), (224,224))
        imgs.append(img)
    mean_img = np.mean(imgs, axis=0)
    fig, ax = plt.subplots()
    ax.imshow(mean_img, cmap="hot")
    ax.set_title(f"Mean Image ({label})")
    save_plot(fig, f"mean_image_{label}.png")

# =============================
# 6. PCA Visualization (Quick)
# =============================
features, labels = [], []
subset = raw_df.sample(min(300, len(raw_df)))
for row in subset.itertuples():
    img = cv2.resize(load_image(row.path), (64,64))
    features.append(img.flatten())
    labels.append(0 if row.label=="Normal" else 1)

X = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="coolwarm", alpha=0.7)
ax.set_title("PCA Projection of Images")
ax.legend(handles=scatter.legend_elements()[0], labels=["Normal","Tuberculosis"])
save_plot(fig, "pca_scatter.png")

# =============================
# Save Stats Summary
# =============================
with open(os.path.join(RESULTS_PATH, "eda_summary.txt"), "w") as f:
    f.write("=== TB-Net Dataset EDA Summary ===\n")
    f.write(f"Raw dataset size: {len(raw_df)}\n")
    f.write(f"Processed dataset size: {len(proc_df)}\n\n")
    f.write("Class distribution (Raw):\n")
    f.write(str(raw_df['label'].value_counts()) + "\n\n")
    f.write("Class distribution (Processed):\n")
    f.write(str(proc_df['label'].value_counts()) + "\n\n")
    f.write("Check eda_results/ folder for plots.\n")

print("✅ EDA Complete. Results saved in 'eda_results/'")
