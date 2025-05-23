from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from collections import defaultdict

def get_dataloaders(iid, num_clients, batch_size, dataset="mnist", classes_per_client=2):
    # === Load Dataset ===
    if dataset == "mnist":
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported")

    labels = np.array(train_dataset.targets)

    # === IID Partition ===
    if iid:
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)

    # === Non-IID Partition (Configurable number of classes per client) ===
    else:
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[int(label)].append(idx)

        for cls in class_indices:
            random.shuffle(class_indices[cls])

        class_pool = list(class_indices.keys())
        split_indices = [[] for _ in range(num_clients)]

        for client_id in range(num_clients):
            selected_classes = random.sample(class_pool, classes_per_client)
            for cls in selected_classes:
                take = random.randint(100, 300)
                cls_samples = class_indices[cls][:take]
                split_indices[client_id].extend(cls_samples)
                class_indices[cls] = class_indices[cls][take:]

            random.shuffle(split_indices[client_id])
            print(f"[Client {client_id}] Classes: {selected_classes}, Samples: {len(split_indices[client_id])}")

    # === Build Dataloaders ===
    dataloaders = []
    for i in range(num_clients):
        subset = Subset(train_dataset, split_indices[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        dataloaders.append(loader)

    return dataloaders
