import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(CNN, self).__init__()
        in_channels = 1 if dataset == 'mnist' else 3
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7 if dataset == 'mnist' else 32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
