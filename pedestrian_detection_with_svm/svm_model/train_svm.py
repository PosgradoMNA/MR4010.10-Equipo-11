"""
SVM Pedestrian Detection Model Training Script

Trains a HOG + Linear SVM classifier for pedestrian detection using the
INRIA Person Dataset (Kaggle version with PASCAL VOC annotations).

References:
- Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005
- INRIA Person Dataset (Kaggle): https://www.kaggle.com/datasets/jcoral02/inriaperson
- scikit-learn LinearSVC documentation

Usage:
    python train_svm.py

Output:
    pedestrian_svm_model.pkl  - Trained SVM model
    pedestrian_svm_scaler.pkl - Feature scaler
    confusion_matrix.png      - Confusion matrix plot
"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = "/Users/cesarivp/Downloads/archive"
TRAIN_IMAGES = os.path.join(DATASET_PATH, "Train", "JPEGImages")
TRAIN_ANNOTATIONS = os.path.join(DATASET_PATH, "Train", "Annotations")

# HOG parameters (standard Dalal & Triggs for 64x128 window)
HOG_WIN_SIZE = (64, 128)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

# Output
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "pedestrian_svm_model.pkl")
SCALER_OUTPUT = os.path.join(os.path.dirname(__file__), "pedestrian_svm_scaler.pkl")


def create_hog_descriptor():
    """Create HOG descriptor with standard pedestrian detection parameters."""
    return cv2.HOGDescriptor(
        HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE, HOG_CELL_SIZE, HOG_NBINS
    )


def extract_hog_features(image, hog):
    """Extract HOG features from a single image resized to 64x128."""
    resized = cv2.resize(image, HOG_WIN_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    features = hog.compute(gray)
    return features.flatten()


def parse_annotation(xml_path):
    """Parse PASCAL VOC XML annotation, return list of bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        if obj.find("name").text == "person":
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def load_positive_samples(hog):
    """Load positive samples by cropping pedestrians from bounding boxes."""
    features = []
    annotations = glob.glob(os.path.join(TRAIN_ANNOTATIONS, "*.xml"))

    for xml_path in annotations:
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        img_path = os.path.join(TRAIN_IMAGES, basename + ".png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = parse_annotation(xml_path)
        for (xmin, ymin, xmax, ymax) in boxes:
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            feat = extract_hog_features(crop, hog)
            features.append(feat)

    print(f"Loaded {len(features)} positive samples")
    return features


def load_negative_samples(hog, max_samples=2500):
    """
    Load negative samples by extracting random patches from positive images
    that do NOT overlap with any pedestrian bounding box.
    """
    features = []
    annotations = glob.glob(os.path.join(TRAIN_ANNOTATIONS, "*.xml"))
    np.random.shuffle(annotations)

    for xml_path in annotations:
        if len(features) >= max_samples:
            break

        basename = os.path.splitext(os.path.basename(xml_path))[0]
        img_path = os.path.join(TRAIN_IMAGES, basename + ".png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if h < 128 or w < 64:
            continue

        boxes = parse_annotation(xml_path)

        # Extract random patches avoiding pedestrian regions
        for _ in range(10):
            if len(features) >= max_samples:
                break
            y = np.random.randint(0, h - 128)
            x = np.random.randint(0, w - 64)

            # Check overlap with any bounding box
            overlaps = False
            for (xmin, ymin, xmax, ymax) in boxes:
                if x < xmax and x + 64 > xmin and y < ymax and y + 128 > ymin:
                    overlaps = True
                    break

            if not overlaps:
                patch = img[y:y+128, x:x+64]
                feat = extract_hog_features(patch, hog)
                features.append(feat)

    print(f"Loaded {len(features)} negative samples")
    return features


def main():
    start_time = time.time()

    hog = create_hog_descriptor()

    print("Loading positive samples (cropping from bounding boxes)...")
    pos_features = load_positive_samples(hog)

    print("Loading negative samples (random non-pedestrian patches)...")
    neg_features = load_negative_samples(hog)

    # Create labels: 1 = pedestrian, 0 = non-pedestrian
    X = np.array(pos_features + neg_features)
    y = np.array([1] * len(pos_features) + [0] * len(neg_features))

    print(f"\nTotal samples: {len(X)} ({len(pos_features)} pos, {len(neg_features)} neg)")
    print(f"Feature vector size: {X.shape[1]}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Linear SVM
    print("\nTraining Linear SVM...")
    svm = LinearSVC(C=0.01, max_iter=10000, random_state=42)
    svm.fit(X_train, y_train)

    # Evaluate
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Pedestrian", "Pedestrian"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Pedestrian", "Pedestrian"])
    disp.plot(cmap="Blues")
    cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.title("SVM Pedestrian Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # Save model and scaler
    joblib.dump(svm, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"\nModel saved to: {MODEL_OUTPUT}")
    print(f"Scaler saved to: {SCALER_OUTPUT}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
