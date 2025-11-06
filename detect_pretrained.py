import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# --- Paths ---
test_dir = "testimages/"         # folder with your images
output_dir = "output_images/"    # where annotated images will be saved

# --- Load model ---
print("⏳ Loading pre-trained COCO model from TensorFlow Hub...")
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(model_url)
print("✅ Model loaded successfully!")

# --- Make sure output folder exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Run detection on all images ---
for fname in os.listdir(test_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read {fname}")
            continue

        # Convert and run detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(img_rgb)[tf.newaxis, ...]
        detections = detector(input_tensor)

        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(np.int32)
        scores = detections["detection_scores"][0].numpy()

        h, w, _ = img.shape
        for i in range(len(scores)):
            if scores[i] > 0.5:  # detection threshold
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f"Object {classes[i]} ({scores[i]*100:.1f}%)"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Save annotated image
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img)
        print(f"🖼️ Saved annotated image: {out_path}")

print("\n✅ Detection complete! Check the 'output_images' folder for results.")
