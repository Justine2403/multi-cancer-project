import torch
from torch import nn
from torchvision import models

from src.data_preprocessing import show_batch, preprocess, download_kaggle_dataset
from src.evaluation import evaluate_model
from src.training import train_model

if __name__ == '__main__':
	dataset, train_loader, val_loader = preprocess()
	show_batch(train_loader, dataset)

	# Définir un modèle pré-entraîné (par exemple, ResNet18)
	model = models.resnet18(pretrained=True)
	# Adapter la dernière couche au nombre de classes
	model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

	# Envoyer le modèle sur le GPU si disponible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()

	num_epochs = 2
	for epoch in range(num_epochs):
		train_loss = train_model(model, dataset, train_loader, val_loader, criterion, device)
		validation_acc = evaluate_model(model, val_loader, criterion, device)
		print(f"\nEpoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {validation_acc:.2f}")
