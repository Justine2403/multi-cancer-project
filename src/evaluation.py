import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def evaluate_model(model, val_loader, criterion, device):
	# Evaluate on the validation set
	model.eval()
	val_loss = 0.0
	correct = 0
	total = 0
	all_preds = []
	all_labels = []

	with torch.no_grad():
		# Use tqdm for progress bar
		progress_bar = tqdm(val_loader, desc="Validation", leave=True)

		for batch_idx, (images, labels) in enumerate(progress_bar):
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			val_loss += loss.item()

			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

			# Collect predictions and true labels for F1 score
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

			# Update progress bar
			progress_bar.set_postfix()

	# Calculate validation accuracy
	val_accuracy = correct / total

	# Calculate F1 score
	f1 = f1_score(all_labels, all_preds, average='weighted')  # Use weighted mean for multi-class

	return val_loss, val_accuracy, f1