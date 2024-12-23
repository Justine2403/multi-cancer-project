import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def loss_function_graph(model_name):
    with open("../results.json", 'r') as file:
        data = json.load(file)

    if model_name not in data:
        print(f"Model '{model_name}' not found in results.json")
        return

    model_data = data[model_name]
    train_losses = model_data.get('train_loss', [])
    val_losses = model_data.get('val_loss', [])

    #print("Train Losses:", train_losses)
    #print("Validation Losses:", val_losses)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss over Epochs for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid()
    plt.show()


def plot_roc_curve(model_name, y_true, y_pred_probs):
    # Vérification de la forme des données
    if len(y_true.shape) == 1:  # Si y_true est un vecteur 1D (classification binaire)
        y_true_bin = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs).reshape(-1, 1)  # Transformation en tableau 2D pour la probabilité de la classe positive
    else:  # Si y_true est déjà un tableau 2D (multi-classe)
        y_true_bin = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)

    n_classes = y_true_bin.shape[1] if len(y_true_bin.shape) > 1 else 2  # Nombre de classes

    # Calcul de la courbe ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)

        # Afficher la courbe ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve class {i} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name} - Class {i}')
        plt.legend(loc='lower right')
        plt.show()



def plot_confusion_matrix(model_name, y_true, y_pred):
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Afficher la matrice de confusion avec Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()