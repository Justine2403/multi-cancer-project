import os
import kaggle

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

KAGGLE_DATASET = "obulisainaren/multi-cancer"
DATA_DIR = "../data/dataset"

def download_kaggle_dataset(force_download=False):
	if not os.path.exists(DATA_DIR):
		kaggle.api.dataset_download_files('obulisainaren/multi-cancer', path=DATA_DIR, unzip=True)
	else:
		print(f"Dataset already exists at {DATA_DIR}")

def preprocess():
	# Define transformations for preprocessing
	transform = transforms.Compose([
		transforms.Resize((224, 224)),  # Resize to 224x224
		transforms.ToTensor(),  # Convert to PyTorch tensor
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization ImageNet
	])

	download_kaggle_dataset()

	# Load data from directory
	dataset = datasets.ImageFolder(root='../data/dataset/Multi Cancer/Multi Cancer', transform=transform)

	# Divide dataset into training (70%) and validation (30%)
	train_size = int(0.7 * len(dataset))  # 70% for training
	val_size = len(dataset) - train_size  # 30% for validation
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	# Create DataLoaders to load the data in batches
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

	# Print some information
	print(f"Nombre d'images dans le dataset total : {len(dataset)}")
	print(f"Nombre d'images pour l'entraînement : {len(train_dataset)}")
	print(f"Nombre d'images pour la validation : {len(val_dataset)}")

	# Print the available classes
	print(f"Classes : {dataset.classes}")

	return dataset, train_loader, val_loader

# Visualization example of a batch's images
def show_batch(loader, dataset):
    images, labels = next(iter(loader))
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Display the 9 first images of the batch
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())  # Convert into a compatible format for matplotlib
        plt.title(dataset.classes[labels[i]])
        plt.axis("off")
    plt.show()