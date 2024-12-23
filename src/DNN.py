import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(DNN, self).__init__()
        
        layers = []
        in_features = input_size

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Dropout to prevent overfitting
            in_features = hidden_size

        # Final output layer
        layers.append(nn.Linear(in_features, num_classes))

        # Combine layers into a sequential block
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
