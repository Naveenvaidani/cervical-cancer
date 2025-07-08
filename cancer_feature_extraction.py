import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

# === STEP 1: PATH TO YOUR DATASET ===
dataset_dir = '.'  # Current directory (since your subfolders are here)


if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"The folder '{dataset_dir}' does not exist.")

# === STEP 2: SETTINGS ===
IMAGE_SIZE = (224, 224)
OUTPUT_CSV = 'features.csv'

# === STEP 3: LOAD MODEL ===
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# === STEP 4: LABEL ENCODING ===
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
print(f"Class to label mapping: {class_to_idx}")

features_list = []
labels_list = []

# === STEP 5: PROCESS IMAGES ===
for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)
    for img_name in tqdm(os.listdir(class_path), desc=f'Processing {class_name}'):
        img_path = os.path.join(class_path, img_name)
        
        try:
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            features = model.predict(img_array, verbose=0).flatten()

            features_list.append(features)
            labels_list.append(class_to_idx[class_name])
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

# === STEP 6: SAVE FEATURES TO CSV ===
features_array = np.array(features_list)
labels_array = np.array(labels_list).reshape(-1, 1)
data_with_labels = np.hstack([features_array, labels_array])

columns = [f'feat_{i}' for i in range(features_array.shape[1])] + ['label']
df = pd.DataFrame(data_with_labels, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Feature extraction complete. CSV saved to: {OUTPUT_CSV}")
