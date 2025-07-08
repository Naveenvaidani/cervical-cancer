import numpy as np
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import joblib
import cv2  # For displaying the image

# === SETTINGS ===
IMAGE_PATH = 'test_image.bmp'  # Replace with your actual image filename
IMAGE_SIZE = (224, 224)

# Use your actual sorted folder names (same order as used during training)
CLASS_NAMES = [
    'dyskeratotic',
    'koilocytotic',
    'metaplastic',
    'parabasal',
    'superficialIntermediate'
]

# === Load the trained model ===
classifier = joblib.load('rf_classifier.pkl')  # <- Updated model filename

# === Load ResNet50 for feature extraction ===
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

# === Load and preprocess the image ===
img = load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
img_array = img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)
img_array_preprocessed = preprocess_input(img_array_expanded)

# === Extract features ===
features = feature_model.predict(img_array_preprocessed, verbose=0)
features_flattened = features.flatten().reshape(1, -1)

# === Predict class ===
prediction = classifier.predict(features_flattened)
predicted_class = CLASS_NAMES[int(prediction[0])]

print(f"\n✅ Prediction Result: The image is classified as **{predicted_class.upper()}**")

# === Display image with prediction using OpenCV ===
img_cv = cv2.imread(IMAGE_PATH)

if img_cv is None:
    print("⚠️ Could not load the image using OpenCV.")
else:
    # Resize for display
    display_img = cv2.resize(img_cv, (400, 400))
    
    # Put the predicted class name on the image
    cv2.putText(display_img, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show image in a window
    cv2.imshow("Cervical Cell Classification", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
