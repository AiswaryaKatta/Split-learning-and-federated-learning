import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_local(model, dataset, indices, epochs, batch_size=16, lr=0.01):
    model.train()
    model = model.to(device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def average_weights(weights):
    avg_weights = copy.deepcopy(weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(weights)):
            avg_weights[key] += weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(weights))
    return avg_weights

def test_model(model, testset):
    model.eval()
    model = model.to(device)
    test_loader = DataLoader(testset, batch_size=64)
    correct, total = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100.0 * correct / total
