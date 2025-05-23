import os
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"

import streamlit as st
import matplotlib.pyplot as plt
from emissions_tracker import CarbonTracker
from split_learning.main import run_split_learning
from federated_learning.run_federated import run_federated_learning
from split_learning.config import config as split_config_full
from multi_emission_charts import draw_emission_graphs
import pandas as pd

import matplotlib.pyplot as plt

def plot_accuracy(acc_list):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(acc_list)+1), acc_list, marker='o', linestyle='-', color='royalblue')
    plt.title("Accuracy vs Rounds", fontsize=18)
    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

def plot_loss(loss_list):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(loss_list)+1), loss_list, marker='x', linestyle='-', color='firebrick')
    plt.title("Loss vs Rounds", fontsize=18)
    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

def plot_time(time_list):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(time_list)+1), time_list, marker='d', linestyle='-', color='goldenrod')
    plt.title("Training Time per Round", fontsize=18)
    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

def plot_emissions(emissions_list):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(emissions_list)+1), emissions_list, marker='s', linestyle='-', color='seagreen')
    plt.title("Carbon Emissions vs Rounds", fontsize=18)
    plt.xlabel("Rounds", fontsize=14)
    plt.ylabel("CO₂ Emitted (kg)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

def show_accuracy_plot(split_acc=None, fed_acc=None):
    if not split_acc and not fed_acc:
        st.warning("⚠️ No accuracy data to plot.")
        return
    plt.figure(figsize=(10, 4))
    if split_acc:
        plt.plot(range(1, len(split_acc)+1), split_acc, marker='o', label='Split Learning')
    if fed_acc:
        plt.plot(range(1, len(fed_acc)+1), fed_acc, marker='x', label='Federated Learning')
    plt.title("Accuracy Comparison Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_samples_vs_emissions(samples_list, emissions_list):
    plt.figure(figsize=(10,6))
    plt.plot(samples_list, emissions_list, marker='o', linestyle='-', color='mediumvioletred')
    plt.title("Carbon Emissions vs Samples Trained", fontsize=18)
    plt.xlabel("Samples Processed", fontsize=14)
    plt.ylabel("CO₂ Emitted (kg)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()


# Streamlit layout
st.set_page_config(page_title="Unified ML Dashboard", layout="wide")
st.title(" Unified ML Training with Emissions Tracker")

tab1, tab2, tab3 = st.tabs(["Split Learning", " Federated Learning", " Simultaneous Run"])

# ========== TAB 1: SPLIT LEARNING ==========
with tab1:
    st.header("Split Learning Settings")

    dataset = st.selectbox("Dataset", ["mnist", "cifar10"])
    num_rounds = st.number_input("Number of Rounds", 1, value=2)
    num_clients = st.number_input("Clients", 1, value=2)
    epochs_per_client = st.number_input("Epochs/Client", 1, value=1)
    iid = st.radio("Data Distribution", ["IID", "Non-IID"]) == "IID"

    classes_per_client = st.slider("Classes per Client (Non-IID only)", 1, 5, value=2) if not iid else 0
    learning_rate = st.number_input("Learning Rate", value=0.01)
    batch_size = st.number_input("Batch Size", value=64)
    optimizer = st.selectbox("Optimizer", ["SGD", "Adam"])
    loss_fn = st.selectbox("Loss Function", ["CrossEntropyLoss"])
    dropout = st.slider("Dropout Rate", 0.0, 0.9, 0.0)
    activation = st.selectbox("Activation", ["relu"])
    communication_frequency = st.number_input("Communication Frequency", value=1)

    if st.button(" Train Split Learning"):
        model_config = split_config_full["model_configs"][dataset]
        config = {
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "epochs_per_client": epochs_per_client,
            "iid": iid,
            "dataset": dataset,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "dropout": dropout,
            "activation": activation,
            "communication_frequency": communication_frequency,
            "client_model_layers": model_config["client_model_layers"],
            "server_model_layers": model_config["server_model_layers"],
            "model_configs": split_config_full["model_configs"],
            "classes_per_client": classes_per_client,
            "metrics": ["accuracy"]
        }

        with CarbonTracker(project_name="Split Learning", output_dir="emissions_logs"):
            acc_list = run_split_learning(config)

        if acc_list:
            st.success(f"✅ Final Accuracy: {acc_list[-1]:.2f}%")
            show_accuracy_plot(split_acc=acc_list)

    if st.button(" Show Detailed Emissions Insights (Split Learning)"):
        f1, f2, f3 = draw_emission_graphs(experiment_type="Split Learning")
        if f1: st.pyplot(f1)
        if f2: st.pyplot(f2)
        if f3: st.pyplot(f3)

# ========== TAB 2: FEDERATED LEARNING ==========
with tab2:
    st.header("Federated Learning Settings")

    dataset = st.selectbox("Dataset", ["mnist", "cifar10"], key="fed_data")
    rounds = st.number_input("Number of Rounds", 1, value=2, key="fed_rounds")
    clients = st.number_input("Clients", 1, value=2, key="fed_clients")
    epochs = st.number_input("Epochs per Client", 1, value=1, key="fed_epochs")
    iid = st.radio("Data Distribution", ["IID", "Non-IID"], key="fed_iid") == "IID"

    if st.button(" Train Federated Learning"):
        fed_config = {
            "dataset": dataset,
            "rounds": rounds,
            "clients": clients,
            "epochs": epochs,
            "iid": iid
        }

        with CarbonTracker(project_name="Federated Learning", output_dir="emissions_logs"):
            acc_history = run_federated_learning(fed_config)

        if acc_history:
            st.success(f"✅ Final Accuracy: {acc_history[-1]:.2f}%")
            show_accuracy_plot(fed_acc=acc_history)

    if st.button(" Show Detailed Emissions Insights (Federated Learning)"):
        f1, f2, f3 = draw_emission_graphs(experiment_type="Federated Learning")
        if f1: st.pyplot(f1)
        if f2: st.pyplot(f2)
        if f3: st.pyplot(f3)

# ========== TAB 3: SIMULTANEOUS RUN ==========
with tab3:
    st.header("Simultaneous Split + Federated Learning")

    st.subheader("Split Learning Settings for Simultaneous Run")
    dataset_split = st.selectbox("Split Dataset", ["mnist", "cifar10"], key="split_simul")
    num_rounds_split = st.number_input("Split: Number of Rounds", 1, value=2, key="split_rounds")
    num_clients_split = st.number_input("Split: Clients", 1, value=2, key="split_clients")
    epochs_split = st.number_input("Split: Epochs per Client", 1, value=1, key="split_epochs")
    iid_split = st.radio("Split: Data Distribution", ["IID", "Non-IID"], key="split_iid") == "IID"

    st.subheader("Federated Learning Settings for Simultaneous Run")
    dataset_fed = st.selectbox("Federated Dataset", ["mnist", "cifar10"], key="fed_simul")
    num_rounds_fed = st.number_input("Federated: Number of Rounds", 1, value=2, key="fed_simul_rounds")
    num_clients_fed = st.number_input("Federated: Clients", 1, value=2, key="fed_simul_clients")
    epochs_fed = st.number_input("Federated: Epochs per Client", 1, value=1, key="fed_simul_epochs")
    iid_fed = st.radio("Federated: Data Distribution", ["IID", "Non-IID"], key="fed_simul_iid") == "IID"

    run_both = st.button(" Train Both Simultaneously")

    if run_both:
        # Split Learning first
        model_config = split_config_full["model_configs"][dataset_split]
        with CarbonTracker(project_name="Split Learning", output_dir="emissions_logs"):
            st.write(" Training Split Learning...")
            split_config = {
                "num_rounds": num_rounds_split,
                "num_clients": num_clients_split,
                "epochs_per_client": epochs_split,
                "iid": iid_split,
                "dataset": dataset_split,
                "learning_rate": 0.01,
                "batch_size": 64,
                "optimizer": "SGD",
                "loss_fn": "CrossEntropyLoss",
                "dropout": 0.0,
                "activation": "relu",
                "communication_frequency": 1,
                "client_model_layers": model_config["client_model_layers"],
                "server_model_layers": model_config["server_model_layers"],
                "model_configs": split_config_full["model_configs"],
                "classes_per_client": 2,
                "metrics": ["accuracy"]
            }
            split_acc = run_split_learning(split_config)

        # Federated Learning second
        with CarbonTracker(project_name="Federated Learning", output_dir="emissions_logs"):
            st.write(" Training Federated Learning...")
            fed_config = {
                "dataset": dataset_fed,
                "rounds": num_rounds_fed,
                "clients": num_clients_fed,
                "epochs": epochs_fed,
                "iid": iid_fed
            }
            fed_acc = run_federated_learning(fed_config)

        st.success("✅ Both trainings completed.")
        show_accuracy_plot(split_acc=split_acc, fed_acc=fed_acc)

    if st.button(" Show Detailed Emissions Insights (Both Experiments)"):
        f1, f2, f3 = draw_emission_graphs(experiment_type=None)  # No filter
        if f1: st.pyplot(f1)
        if f2: st.pyplot(f2)
        if f3: st.pyplot(f3)
