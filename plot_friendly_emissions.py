import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_friendly_emissions(csv_path="emissions_logs/friendly_emissions.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠️ 'friendly_emissions.csv' is empty.")
        return None

    print("📄 Columns found:", list(df.columns))

    date_col = "Date"
    exp_col = "Type of Experiment"
    carbon_col = "Carbon Emitted (kg)"

    if not all(col in df.columns for col in [date_col, exp_col, carbon_col]):
        print("🛑 Required columns missing!")
        return None

    df[date_col] = pd.to_datetime(df[date_col])

    # Soft legend names if typos exist
    df[exp_col] = df[exp_col].str.replace("Learninging", "Learning")

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_type in df[exp_col].unique():
        df_sub = df[df[exp_col] == exp_type]
        ax.plot(df_sub[date_col], df_sub[carbon_col], marker='o', linestyle='-', markersize=6, label=exp_type)

    ax.set_title("Carbon Emissions Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Carbon Emitted (kg)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Rotate dates better
    plt.xticks(rotation=45)
    plt.tight_layout()

    ax.legend(title="Experiment Type", fontsize=10, title_fontsize=11)
    return fig
