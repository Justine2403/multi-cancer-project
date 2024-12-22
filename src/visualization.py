import matplotlib.pyplot as plt
import json


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