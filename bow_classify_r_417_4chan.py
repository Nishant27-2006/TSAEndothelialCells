import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_multi_channel_images_from_folder(folder):
    images = []
    labels = []  # Labels should be 0 (no nitric oxide) or 1 (nitric oxide present)
    for filename in os.listdir(folder):
        # Load multi-channel image - update reading method if different channels are stored differently
        img = cv2.imread(os.path.join(folder, filename))
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

def extract_features_from_multi_channel_images(images):
    # Implement feature extraction for multi-channel images
    # This could include processing each channel separately or combining them in some way
    features = []
    for img in images:
        # Example: Flatten each channel and concatenate
        feature = np.concatenate([channel.flatten() for channel in cv2.split(img)])
        features.append(feature)
    return features

# Path to the folder containing your multi-channel endothelial cell images
folder = 'path_to_your_image_data'  # Update this path with the actual path to your images
images, labels = load_multi_channel_images_from_folder(folder)
features = extract_features_from_multi_channel_images(images)

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
