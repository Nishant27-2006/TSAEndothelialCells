import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder):
    images = []
    labels = []  # Labels should be 0 (no nitric oxide) or 1 (nitric oxide present)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            # Extract label for each image - implement this based on your labeling scheme
            label = extract_label(filename)  
            labels.append(label)
    return images, labels

def extract_label(filename):
    # Implement this function to extract labels from filenames or a corresponding metadata file
    # Placeholder: returns 1 if 'no' (indicating nitric oxide) is not in the filename
    return int('no' not in filename)

def extract_features(images):
    # Implement feature extraction here
    # Example: Flatten the image or apply more complex feature extraction methods
    features = [img.flatten() for img in images]
    return features

# Path to the folder containing your endothelial cell images
folder = 'path_to_your_image_data'  # Update this path with the actual path to your images
images, labels = load_images_from_folder(folder)
features = extract_features(images)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Setting up the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
