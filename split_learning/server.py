import torch

class Server:
    def __init__(self, model_class):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class().to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
