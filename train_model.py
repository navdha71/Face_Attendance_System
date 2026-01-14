import tensorflow as tf                     # Import TensorFlow library for deep learning
from tensorflow.keras import layers, models  # Import layers and model tools from Keras
import cv2                                 # Import OpenCV for image reading and processing
import numpy as np                         # Import NumPy for numerical operations
import os                              # Import OS module to work with files and folders

data = []                                # List to store image data
labels = []                              # List to store labels for each image
label_dict = {}                          # Dictionary to map label numbers to person names

dataset_path = "dataset"                # Dictionary to map label numbers to person names
label = 0                               # Initialize label number starting from 0

print("Loading images...")              # Print message to show image loading has started

for person in os.listdir(dataset_path):    # Loop through each person folder inside dataset
    person_path = os.path.join(dataset_path, person) 
                                           # Create full path for each person folder

    if not os.path.isdir(person_path):     # Check if the path is NOT a folder
        continue                         # Skip it and move to next item

    label_dict[label] = person           # Store person name with its label number

    for img in os.listdir(person_path):       # Loop through each image in person’s folder
        img_path = os.path.join(person_path, img)   # Create full path for the image file

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
                                                       # Read the image in grayscale format
        image = cv2.resize(image, (100, 100))       # Resize image to 100x100 pixels

        data.append(image)                     # Add the processed image to data list
        labels.append(label)                   # Add corresponding label to labels list

    label += 1                        # Increase label number for next person

data = np.array(data).reshape(-1, 100, 100, 1) / 255.0 
                                               # Convert data list to NumPy array
                                               # Reshape for CNN input
                                               # Normalize pixel values (0–255 → 0–1)
labels = np.array(labels)                 # Convert labels list into NumPy array

print("Building CNN model...")              # Print message before creating CNN model

model = models.Sequential([                       # Create a Sequential CNN model
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
                                                  # First convolution layer
                                                  # 32 filters of size 3x3
                                                  # ReLU activation
                                                  # Input shape for grayscale image
    layers.MaxPooling2D(2,2),             # Reduce image size using max pooling

    layers.Conv2D(64, (3,3), activation='relu'), # Second convolution layer with 64 filters
    layers.MaxPooling2D(2,2),               # Another max pooling layer

    layers.Flatten(),                    # Convert 2D feature maps into 1D vector
    layers.Dense(128, activation='relu'),  # Fully connected hidden layer with 128 neurons
    layers.Dense(len(label_dict), activation='softmax') 
                                                # Output layer
                                                # Number of neurons = number of people
                                                # Softmax gives probability for each person
])

model.compile(
    optimizer='adam',                          # Use Adam optimizer for faster training
    loss='sparse_categorical_crossentropy',   # Loss function for multi-class classification
    metrics=['accuracy']                    # Measure accuracy during training
)

print("Training model...")                    # Print message before training starts
model.fit(data, labels, epochs=10)            # Train the CNN model for 10 epochs

model.save("model/face_model.h5")               # Save trained model to disk
np.save("model/labels.npy", label_dict)        # Save label dictionary for later prediction

print("Model training completed and saved")       # Print completion message
