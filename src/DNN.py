import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(DNN, self).__init__()

        layers = []
        in_features = input_size

        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Dropout to reduce overfitting
            in_features = hidden_size

        # Output layer
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        return self.model(x)
