import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os

# ----------------------------
# 1. Paths
# ----------------------------
dataset_dir = "dataset/"  # your dataset folder
test_image_path = "test.jpg"  # image to test

# ----------------------------
# 2. Data generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# ----------------------------
# 3. Load pre-trained MobileNetV2
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base

# Add classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# 4. Train the model
# ----------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10  # start small for testing
)

# Save the trained model
model.save("plastic_detector_model.h5")
print("✅ Model saved successfully!")

# ----------------------------
# 5. Test on new image
# ----------------------------
img = cv2.imread(test_image_path)
if img is None:
    print("❌ Test image not found")
else:
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        print("Plastic detected ✅")
    else:
        print("Not plastic ❌")
