import os
import cv2
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from skimage import feature
import kaggle

KAGGLE_DATASET = "obulisainaren/multi-cancer"
DATA_DIR = "../data/dataset"

def download_kaggle_dataset(force_download=False):
    if not os.path.exists(DATA_DIR) or force_download:
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
    else:
        print(f"Dataset already exists at {DATA_DIR}")

def extract_lbp_features(image, numPoints=8, radius=2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(img_gray, numPoints, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    return hist / hist.sum()

def load_data_with_lbp(dataset):
    features = []
    labels = []
    for img, label in dataset:
        # Convertir le tensor en image numpy
        img_np = np.array(img.permute(1, 2, 0) * 255, dtype=np.uint8)
        lbp_features = extract_lbp_features(img_np)
        features.append(lbp_features)
        labels.append(label)
    return np.array(features), np.array(labels)

def preprocess():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    download_kaggle_dataset()
    dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "Multi Cancer/Multi Cancer"), transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset, dataset

def main():
    train_dataset, val_dataset, dataset = preprocess()

    print("Extracting LBP features for training data...")
    X_train, y_train = load_data_with_lbp(train_dataset)
    print("Extracting LBP features for validation data...")
    X_test, y_test = load_data_with_lbp(val_dataset)

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_test.shape}, Labels shape: {y_test.shape}")

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=dataset.classes))

if __name__ == "__main__":
    main()