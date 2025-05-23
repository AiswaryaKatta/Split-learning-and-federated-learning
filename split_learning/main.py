import torch
import copy
import streamlit as st

def run_split_learning(config):
    from split_learning.client import Client
    from split_learning.server import Server
    from split_learning.data_utils import get_dataloaders
    from split_learning.train_utils import train_client, evaluate
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    import time

    def get_loss_fn(name):
        return getattr(nn, name)()

    def get_optimizer(opt_name, params, lr):
        if opt_name.lower() == "sgd":
            return optim.SGD(params, lr=lr)
        elif opt_name.lower() == "adam":
            return optim.Adam(params, lr=lr)

    dataset_name = config["dataset"]
    client_class = config["model_configs"][dataset_name]["client_model_layers"]
    server_class = config["model_configs"][dataset_name]["server_model_layers"]

    clients = [Client(client_class) for _ in range(config["num_clients"])]
    server = Server(server_class)

    transform = transforms.ToTensor()
    if dataset_name == "mnist":
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("./data", train=False, download=True, transform=transform),
            batch_size=1000, shuffle=False
        )
    elif dataset_name == "cifar10":
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("./data", train=False, download=True, transform=transform),
            batch_size=1000, shuffle=False
        )
    else:
        raise ValueError("Unsupported dataset")

    data_loaders = get_dataloaders(
        config["iid"],
        config["num_clients"],
        config["batch_size"],
        dataset=dataset_name,
        classes_per_client=config.get("classes_per_client", 2)
    )

    loss_fn = get_loss_fn(config["loss_fn"])
    acc_list = []

    # ✅ Add Streamlit Progress Bar
    progress_bar = st.progress(0)
    stop_training = st.button("🛑 Stop Training Early")

    print("\n[🚀 Split Learning Training Begins]\n")

    for rnd in range(config["num_rounds"]):
        if stop_training:
            st.warning(f"Training manually stopped at Round {rnd+1}")
            break

        print(f"\n--- Round {rnd+1}/{config['num_rounds']} ---")
        round_start = time.time()

        client_weights = []
        server_weights = []

        for i in range(config["num_clients"]):
            client = clients[i]
            data_len = len(data_loaders[i].dataset)
            print(f"[👤 Client {i}] - Training on {data_len} samples")

            optimizer = get_optimizer(
                config["optimizer"],
                list(client.model.parameters()) + list(server.model.parameters()),
                config["learning_rate"]
            )

            train_client(client.model, server.model, data_loaders[i], optimizer, loss_fn, config["epochs_per_client"])

            client_weights.append(copy.deepcopy(client.model.state_dict()))
            server_weights.append(copy.deepcopy(server.model.state_dict()))

        # ✅ Aggregation step
        new_client_weights = average_weights(client_weights)
        new_server_weights = average_weights(server_weights)

        for client in clients:
            client.model.load_state_dict(new_client_weights)
        server.model.load_state_dict(new_server_weights)

        # ✅ Evaluate
        acc = evaluate(clients[0].model, server.model, test_loader)
        acc_list.append(acc)

        round_time = time.time() - round_start
        print(f"[📈 Round {rnd+1}] Accuracy: {acc:.2f}% | Time: {round_time:.2f}s")

        # ✅ Update Progress Bar
        progress_bar.progress((rnd + 1) / config["num_rounds"])

    st.success(f"✅ Training complete. Final Accuracy: {acc_list[-1]:.2f}%")
    print(f"\n[✅ Training complete. Final Accuracy: {acc_list[-1]:.2f}%]\n")
    return acc_list

def average_weights(weight_list):
    """Helper function to average a list of model weights"""
    avg = copy.deepcopy(weight_list[0])
    for key in avg.keys():
        for i in range(1, len(weight_list)):
            avg[key] += weight_list[i][key]
        avg[key] = avg[key] / len(weight_list)
    return avg
