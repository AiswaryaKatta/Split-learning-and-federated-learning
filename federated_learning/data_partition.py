from torchvision import datasets, transforms
import numpy as np
import random
from collections import defaultdict

def load_dataset(dataset_name):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    return trainset, testset

def partition_iid(trainset, num_clients):
    data_per_client = int(len(trainset) / num_clients)
    indices = np.random.permutation(len(trainset))
    client_data = {i: indices[i * data_per_client : (i + 1) * data_per_client] for i in range(num_clients)}
    return client_data

def partition_noniid(trainset, num_clients):
    client_data = {i: [] for i in range(num_clients)}
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(trainset):
        label = int(label) if not isinstance(label, int) else label
        class_indices[label].append(idx)

    for cls in class_indices:
        random.shuffle(class_indices[cls])

    class_ptr = {cls: 0 for cls in class_indices}
    classes = list(class_indices.keys())

    for i in range(num_clients):
        chosen_classes = np.random.choice(classes, 2, replace=False)
        assigned = []

        for cls in chosen_classes:
            start = class_ptr[cls]
            take = random.randint(100, 350)
            end = min(start + take, len(class_indices[cls]))
            if start < end:
                assigned += class_indices[cls][start:end]
                class_ptr[cls] = end

        if len(assigned) == 0:
            assigned = random.sample(range(len(trainset)), 100)

        random.shuffle(assigned)
        client_data[i] = assigned

    return client_data

def partition_data(trainset, num_clients=5, iid=True):
    return partition_iid(trainset, num_clients) if iid else partition_noniid(trainset, num_clients)
