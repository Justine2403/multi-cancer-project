import torch
import torch.nn as nn
import torch.nn.functional as F

# remplacer la ligne model du main.py par
# model = CNN(num_classes=len(dataset.classes))
# et enlever la ligne suivante :
# model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_fc1(num_classes)

    def _initialize_fc1(self, num_classes):
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 224, 224)
            sample_output = self.forward_features(sample_input)
            flattened_size = sample_output.shape[1]
            self.fc1 = nn.Linear(flattened_size, 128)

    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x