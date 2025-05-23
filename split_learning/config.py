from split_learning.model_parts import MNISTClient, MNISTServer, CIFARClient, CIFARServer

config = {
    # Split Learning training settings
    "num_rounds": 5,
    "num_clients": 3,
    "epochs_per_client": 1,
    "iid": True,
    "dataset": "mnist",  # "mnist" or "cifar10"

    # Model structure (direct classes instead of dynamic list)
    "model_configs": {
        "mnist": {
            "client_model_layers": MNISTClient,
            "server_model_layers": MNISTServer
        },
        "cifar10": {
            "client_model_layers": CIFARClient,
            "server_model_layers": CIFARServer
        }
    },

    # Training hyperparameters
    "learning_rate": 0.01,
    "batch_size": 64,
    "optimizer": "SGD",
    "loss_fn": "CrossEntropyLoss",
    "dropout": 0.0,
    "activation": "relu",
    "communication_frequency": 1,
    "metrics": ["accuracy"]
}
