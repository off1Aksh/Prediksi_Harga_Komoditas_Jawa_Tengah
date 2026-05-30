import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "..", "Dataset Final", "dataset_harga_komoditas_2021_2026.csv")
scaler_output_path = os.path.join(script_dir, "scaler.gz")

print(f"Loading dataset from: {os.path.abspath(dataset_path)}")

# Read dataset
df = pd.read_csv(dataset_path)

# 1. Convert data types
df["tanggal"] = pd.to_datetime(df["tanggal"], format="%d/ %m/ %Y")
kolom_harga = [col for col in df.columns if col != "tanggal"]
for col in kolom_harga:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# 2. LOCF - Forward Fill missing dates (same as pre-processing.py)
df = df.set_index("tanggal")
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
df = df.reindex(full_range)
df = df.ffill()

# 3. Fit Min-Max Scaler
print("Fitting MinMaxScaler...")
scaler = MinMaxScaler()
scaler.fit(df.values)

# 4. Save the scaler
joblib.dump(scaler, scaler_output_path)
print(f"Scaler successfully saved to: {os.path.abspath(scaler_output_path)}")
