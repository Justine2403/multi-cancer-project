import os
import json
import torch
from torch import nn
from torchvision import models
import numpy as np
from src.visualization import plot_confusion_matrix
from CNN import CNN
from DNN import DNN
from data_preprocessing import show_batch, preprocess
from evaluation import evaluate_model
from training import train_model
from visualization import loss_function_graph


MODEL_DIR = "../models/"
RESULTS_PATH = "../results.json"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_SIZE = 224 * 224 * 3
HIDDEN_SIZE = [512, 256, 128]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def update_json(result_path, results):
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    dataset, train_loader, val_loader = preprocess()
    show_batch(train_loader, dataset)

    MODELS = {
        "resnet18": models.resnet18(pretrained=True),
        "CNN": CNN(num_classes=len(dataset.classes)),
        "DNN": DNN(INPUT_SIZE, HIDDEN_SIZE, num_classes=len(dataset.classes))
    }

    # Prepare models
    for model_name, model in MODELS.items():
        # Verify if the model is an instance of a predefined torchvision model
        if isinstance(model, (models.ResNet, models.VGG)):
            model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    criterion = nn.CrossEntropyLoss()

    # Load global results (or initialize if doesn't exist)
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for model_name, model in MODELS.items():
        model = model.to(device)
        model_dir = os.path.join(MODEL_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Initialize the model's section if not already in the json file
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

        # Print the results or calculate missing values if the model has already been trained
        if results[model_name]["train_complete"] and results[model_name]["validation_complete"]:
            print(f"\nModel {model_name} already trained.")
            print(f"Results: {results[model_name]}")

            # Loss functions display
            loss_function_graph(model_name)

            # Load model for evaluation
            epoch_model_path = os.path.join(model_dir, f"{model_name}_epoch_{len(results[model_name]['epochs'])}.pth")
            if os.path.exists(epoch_model_path):
                print(f"Loading model {model_name} from epoch {len(results[model_name]['epochs'])}...")

                # Load the model on the CPU
                model.load_state_dict(torch.load(epoch_model_path, map_location=torch.device('cpu'), weights_only=True))

                # Get the true labels and predictions of the model
                y_true = []  # List for true labels
                y_pred = []  # List for predictions
                y_pred_probs = []  # List for predicted probabilities

                # Loop to get results for each batch
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())
                        y_pred_probs.extend(probs[:, 1].cpu().numpy())  # Suppose a binary classification

                # Display confusion matrix
                plot_confusion_matrix(model_name, np.array(y_true), np.array(y_pred))
            else:
                print(f"Model for epoch {len(results[model_name]['epochs'])} not found for {model_name}. Skipping...")

            continue  # Go to the next model



        elif results[model_name]["train_complete"] and not results[model_name]["validation_complete"]:
            epochs = len(results[model_name]["epochs"])
            for epoch in range(epochs):
                epoch_model_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch+1}.pth")
                if os.path.exists(epoch_model_path):
                    print(f"\nEvaluating model at epoch {epoch+1} for {model_name}...")
                    model.load_state_dict(torch.load(epoch_model_path))
                    validation_loss, validation_acc, f1_score = evaluate_model(model, val_loader, criterion, device)
                    print(f"Epoch {epoch+1} - Validation Loss: {validation_loss}, Accuracy: {validation_acc}, F1 Score: {f1_score}")
                    # Updating results
                    results[model_name]['val_loss'].append(validation_loss)
                    results[model_name]['val_acc'].append(validation_acc)
                    results[model_name]['f1_score'].append(f1_score)
                    update_json(RESULTS_PATH, results)
                    loss_function_graph(model_name)
                else:
                    print(f"Model for epoch {epoch+1} not found for {model_name}. Skipping...")
            results[model_name]['validation_complete'] = True
            update_json(RESULTS_PATH, results)

            continue  # Go to the next model

        print(f"\nTraining model: {model_name}")

        best_acc = results[model_name]['best_acc']
        num_epochs = 4

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {model_name}")

            train_loss = train_model(model, dataset, train_loader, val_loader, criterion, device)
            validation_loss, validation_acc, f1_score = evaluate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {validation_acc}")

            # Save model after each epoch
            epoch_model_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_model_path)

            # Update results
            results[model_name]['epochs'].append(epoch + 1)
            results[model_name]['train_loss'].append(train_loss)
            results[model_name]['val_loss'].append(validation_loss)
            results[model_name]['val_acc'].append(validation_acc)
            results[model_name]['f1_score'].append(f1_score)

            if validation_acc > best_acc:
                best_acc = validation_acc
                results[model_name]['best_acc'] = best_acc
                print(f"New best accuracy for {model_name} (Accuracy: {best_acc})")

            # Update results.json file
            update_json(RESULTS_PATH, results)

        # Mark the training as completed
        results[model_name]['train_complete'] = True
        results[model_name]['validation_complete'] = True
        update_json(RESULTS_PATH, results)
        loss_function_graph(model_name)

        print(f"\nTraining complete for {model_name}. Results saved.")