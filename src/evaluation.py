import torch
from tqdm import tqdm

def evaluate_model(model, val_loader, criterion, device):
	# Évaluer sur l'ensemble de validation
	model.eval()
	val_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		# Utiliser tqdm pour la barre de progression
		progress_bar = tqdm(val_loader, desc="Validation", leave=True)

		for batch_idx, (images, labels) in enumerate(progress_bar):
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			val_loss += loss.item()

			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

			# Mettre à jour la barre de progression
			progress_bar.set_postfix()

	return correct / total