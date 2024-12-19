import torch
from tqdm import tqdm

def train_model(model, dataset, train_loader, val_loader, criterion, device):
	# Définir l'optimiseur
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# Boucle d'entraînement
	model.train()
	train_loss = 0.0

	# Utiliser tqdm pour la barre de progression
	progress_bar = tqdm(train_loader, desc="Training", leave=True)

	for batch_idx, (images, labels) in enumerate(progress_bar):
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()

		# Mettre à jour la barre de progression avec la perte courante
		progress_bar.set_postfix(loss=loss.item())

	return train_loss / len(train_loader)