import pandas as pd
import time
import os

# Paths
raw_path = "emissions_logs/emissions.csv"
pretty_path = "emissions_logs/pretty_emissions.csv"
friendly_path = "emissions_logs/friendly_emissions.csv"

# === Wait if file is locked ===
while True:
    try:
        with open(raw_path, "r+"):
            break
    except PermissionError:
        print("⏳ Waiting for 'emissions.csv' to be closed...")
        time.sleep(1)

print("📥 Reading emissions.csv...")

# === Load raw emissions file ===
df = pd.read_csv(raw_path)

# === Check if it's empty ===
if df.empty:
    print("⚠️ 'emissions.csv' is empty. No data to convert.")
    exit()

# === Clean + rename columns ===
clean_df = pd.DataFrame({
    "Date": pd.to_datetime(df["timestamp"]).dt.date,
    "Experiment": df["project_name"].str.replace("Split Learn", "Split Learning"),
    "Duration (sec)": df["duration"].round(2),
    "Carbon (kg CO₂)": df["emissions"].round(8),
    "Electricity (kWh)": df["energy_consumed"].round(6),
})

# === Save the cleaned CSV ===
clean_df.to_csv(friendly_path, index=False)
print(f"✅ Cleaned CSV saved as: {friendly_path}")

# === Delete old CSVs ===
for old_path in [raw_path, pretty_path]:
    if os.path.exists(old_path):
        os.remove(old_path)
        print(f"🗑️ Deleted: {old_path}")
