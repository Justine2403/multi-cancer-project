import os
import json
import torch
from torch import nn
from torchvision import models

from src.CNN import CNN
from src.data_preprocessing import show_batch, preprocess
from src.evaluation import evaluate_model
from src.training import train_model
from src.visualization import loss_function_graph

MODEL_DIR = "../models/"
RESULTS_PATH = "../results.json"
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_CLASSES = 10  # À ajuster pour ton dataset

MODELS = {
    "resnet18": models.resnet18(pretrained=True),
    "CNN": CNN(NUM_CLASSES)
}


# Préparer les modèles
for model_name, model in MODELS.items():
    # Vérifie si le modèle est une instance d'un modèle prédéfini de torchvision
    if isinstance(model, (models.ResNet, models.VGG)):
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def update_json(result_path, results):
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    dataset, train_loader, val_loader = preprocess()
    show_batch(train_loader, dataset)

    criterion = nn.CrossEntropyLoss()

    # Charger les résultats globaux (ou initialiser s'il n'existe pas)
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for model_name, model in MODELS.items():
        model = model.to(device)
        model_dir = os.path.join(MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Initialiser la section du modèle s'il n'est pas encore dans le json
        if model_name not in results:
            results[model_name] = {
                "train_complete": False,
                "validation_complete": False,
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "val_acc": [],
                "f1_score": [],
                "best_acc": 0.0
            }

        # Si le modèle est déjà entraîné, afficher les résultats et passer ou calculer valeurs manquantes
        if results[model_name]["train_complete"] and results[model_name]["validation_complete"]:
            print(f"\nModel {model_name} already trained.")
            print(f"Results: {results[model_name]}")
            loss_function_graph(model_name)
            continue    # Passer au modèle suivant
        elif results[model_name]["train_complete"] and not results[model_name]["validation_complete"]:
            epochs = len(results[model_name]["epochs"])
            for epoch in range(epochs):
                epoch_model_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch+1}.pth")
                if os.path.exists(epoch_model_path):
                    print(f"\nEvaluating model at epoch {epoch+1} for {model_name}...")
                    model.load_state_dict(torch.load(epoch_model_path))
                    validation_loss, validation_acc, f1_score = evaluate_model(model, val_loader, criterion, device)
                    print(f"Epoch {epoch+1} - Validation Loss: {validation_loss}, Accuracy: {validation_acc}, F1 Score: {f1_score}")
                    # Mise à jour des résultats
                    results[model_name]['val_loss'].append(validation_loss)
                    results[model_name]['val_acc'].append(validation_acc)
                    results[model_name]['f1_score'].append(f1_score)
                    update_json(RESULTS_PATH, results)
                    loss_function_graph(model_name)
                else:
                    print(f"Model for epoch {epoch+1} not found for {model_name}. Skipping...")
            results[model_name]['validation_complete'] = True
            update_json(RESULTS_PATH, results)

            continue  # Passer au modèle suivant

        print(f"\nTraining model: {model_name}")

        best_acc = results[model_name]['best_acc']
        num_epochs = 4

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {model_name}")

            train_loss = train_model(model, dataset, train_loader, val_loader, criterion, device)
            validation_loss, validation_acc, f1_score = evaluate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {validation_acc}")

            # Sauvegarde du modèle après chaque epoch
            epoch_model_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_model_path)

            # Mise à jour des résultats
            results[model_name]['epochs'].append(epoch + 1)
            results[model_name]['train_loss'].append(train_loss)
            results[model_name]['val_loss'].append(validation_loss)
            results[model_name]['val_acc'].append(validation_acc)
            results[model_name]['f1_score'].append(f1_score)

            if validation_acc > best_acc:
                best_acc = validation_acc
                results[model_name]['best_acc'] = best_acc
                print(f"New best accuracy for {model_name} (Accuracy: {best_acc})")

            # Mise à jour continue du fichier results.json
            update_json(RESULTS_PATH, results)

        # Marquer l'entraînement comme terminé
        results[model_name]['train_complete'] = True
        results[model_name]['validation_complete'] = True
        update_json(RESULTS_PATH, results)
        loss_function_graph(model_name)

        print(f"\nTraining complete for {model_name}. Results saved.")