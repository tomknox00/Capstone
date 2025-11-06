import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# ----------------------------
# Paths
# ----------------------------
model_path = "plastic_detector_model.h5"   # saved model
test_images_dir = "testimages/"           # folder with images to test

# ----------------------------
# Load the saved model
# ----------------------------
if not os.path.exists(model_path):
    print(f"❌ Saved model not found at '{model_path}'. Train the model first.")
    exit()

model = load_model(model_path)
print("✅ Model loaded successfully!")

# ----------------------------
# Test multiple images
# ----------------------------
if not os.path.exists(test_images_dir):
    print(f"❌ Folder '{test_images_dir}' not found. Create it and add images to test.")
else:
    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_images_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not read {filename}")
                continue

            # Resize and normalize
            img_resized = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(img_resized / 255.0, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            label = "Plastic ✅" if prediction[0][0] > 0.5 else "Not plastic ❌"

            print(f"{filename}: {label}")

            # Display image with label
            display_img = img.copy()
            cv2.putText(display_img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prediction[0][0] > 0.5 else (0, 0, 255), 2)
            cv2.imshow(filename, display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
