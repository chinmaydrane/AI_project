import os
import random
import pandas as pd

# Paths
data_folder = "data"  # folder containing all images
output_folder = "."   # where CSVs will be saved

# Get list of images
all_images = os.listdir(data_folder)

# Separate Normal and Tuberculosis images
normal_images = [f for f in all_images if "Normal" in f]
tb_images = [f for f in all_images if "Tuberculosis" in f]

# Shuffle
random.shuffle(normal_images)
random.shuffle(tb_images)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

def split_and_label(images, label):
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = [(img, label) for img in images[:train_end]]
    val = [(img, label) for img in images[train_end:val_end]]
    test = [(img, label) for img in images[val_end:]]
    return train, val, test

# Split Normal
normal_train, normal_val, normal_test = split_and_label(normal_images, 0)
# Split TB
tb_train, tb_val, tb_test = split_and_label(tb_images, 1)

# Combine
train_split = normal_train + tb_train
val_split = normal_val + tb_val
test_split = normal_test + tb_test

# Shuffle combined sets
random.shuffle(train_split)
random.shuffle(val_split)
random.shuffle(test_split)

# Save CSVs
pd.DataFrame(train_split).to_csv(os.path.join(output_folder, "train_split.csv"), index=False, header=False)
pd.DataFrame(val_split).to_csv(os.path.join(output_folder, "val_split.csv"), index=False, header=False)
pd.DataFrame(test_split).to_csv(os.path.join(output_folder, "test_split.csv"), index=False, header=False)

print("✅ CSV files generated successfully!")
print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
