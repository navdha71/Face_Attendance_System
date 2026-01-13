import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os

data = []
labels = []
label_dict = {}

dataset_path = "dataset"
label = 0

print("Loading images...")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_dict[label] = person

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))

        data.append(image)
        labels.append(label)

    label += 1

data = np.array(data).reshape(-1, 100, 100, 1) / 255.0
labels = np.array(labels)

print("Building CNN model...")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_dict), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")
model.fit(data, labels, epochs=10)

model.save("model/face_model.h5")
np.save("model/labels.npy", label_dict)

print("Model training completed and saved")
