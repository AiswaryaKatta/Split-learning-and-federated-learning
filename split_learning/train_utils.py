import torch

def train_client(client_model, server_model, train_loader, optimizer, loss_fn, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model.train()
    server_model.train()

    client_model.to(device)
    server_model.to(device)

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Split Learning Forward Pass
            split_output = client_model(data)
            output = server_model(split_output)

            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()

def evaluate(client_model, server_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model.eval()
    server_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            split_output = client_model(data)
            output = server_model(split_output)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy
