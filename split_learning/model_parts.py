import torch.nn as nn

# ========================================
# Model Parts for Split Learning (CIFAR-10 and MNIST Support)
# ========================================

# ----------- For MNIST (Simple MLP) -----------

class MNISTClient(nn.Module):
    def __init__(self):
        super(MNISTClient, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

class MNISTServer(nn.Module):
    def __init__(self):
        super(MNISTServer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.classifier(x)

# ----------- For CIFAR-10 (Simple CNN) -----------

class CIFARClient(nn.Module):
    def __init__(self):
        super(CIFARClient, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x8x8
        )

    def forward(self, x):
        return self.features(x)

class CIFARServer(nn.Module):
    def __init__(self):
        super(CIFARServer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(x)
