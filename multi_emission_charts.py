import pandas as pd
import matplotlib.pyplot as plt
import os

def draw_emission_graphs(csv_path="emissions_logs/friendly_emissions.csv", experiment_type=None):
    if not os.path.exists(csv_path):
        print(f"❌ friendly_emissions.csv not found.")
        return None, None, None

    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠️ CSV is empty.")
        return None, None, None

    # Rename & clean up columns
    df.columns = [col.strip() for col in df.columns]
    df["Type of Experiment"] = df["Type of Experiment"].str.replace("Learninging", "Learning")

    # ✅ If experiment_type filter is applied
    if experiment_type:
        df = df[df["Type of Experiment"] == experiment_type]
        if df.empty:
            print(f"⚠️ No data found for {experiment_type}")
            return None, None, None

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig3, ax3 = plt.subplots(figsize=(7, 4))

    # 1. Electricity Used vs Carbon Emitted
    for exp in df["Type of Experiment"].unique():
        sub = df[df["Type of Experiment"] == exp]
        ax1.scatter(sub["Electricity Used (kWh)"], sub["Carbon Emitted (kg)"], label=exp)
    ax1.set_title("Carbon Emissions vs Electricity Used")
    ax1.set_xlabel("Electricity Used (kWh)")
    ax1.set_ylabel("Carbon Emitted (kg)")
    ax1.grid(True)
    ax1.legend()

    # 2. Duration vs Carbon Emitted
    for exp in df["Type of Experiment"].unique():
        sub = df[df["Type of Experiment"] == exp]
        ax2.plot(sub["How Long It Ran (in sec)"], sub["Carbon Emitted (kg)"], marker='o', label=exp)
    ax2.set_title("Carbon Emissions vs Duration")
    ax2.set_xlabel("Duration (sec)")
    ax2.set_ylabel("Carbon Emitted (kg)")
    ax2.grid(True)
    ax2.legend()

    # 3. Bar chart: Average Emissions
    avg_emissions = df.groupby("Type of Experiment")["Carbon Emitted (kg)"].mean()
    ax3.bar(avg_emissions.index, avg_emissions.values, color=["orange", "skyblue"])
    ax3.set_title("Average Carbon Emissions by Experiment Type")
    ax3.set_ylabel("Average Carbon Emitted (kg)")
    ax3.grid(axis='y')

    return fig1, fig2, fig3
