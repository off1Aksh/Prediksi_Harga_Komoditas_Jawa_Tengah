import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# direktori
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "dataset_harga_komoditas_2017_2026.csv")

# baca dataset
df = pd.read_csv(input_path)
print(f"Dataset awal: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Tipe data awal:\n{df.dtypes}\n")

# ============================
# 1. Konversi tipe data
# ============================

# konversi kolom tanggal ke datetime
df["tanggal"] = pd.to_datetime(df["tanggal"], format="%d/ %m/ %Y")

# hilangkan koma dan konversi kolom harga ke float
kolom_harga = [col for col in df.columns if col != "tanggal"]
for col in kolom_harga:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

print(f"Tipe data setelah konversi:\n{df.dtypes}\n")

# ============================
# 2. LOCF — isi tanggal kosong
# ============================

# set tanggal sebagai index
df = df.set_index("tanggal")

# buat range tanggal lengkap (harian)
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
missing_dates = full_range.difference(df.index)
print(f"Tanggal tersedia: {len(df)}")
print(f"Tanggal missing (libur/weekend): {len(missing_dates)}")

# reindex agar semua tanggal ada, tanggal baru akan memiliki NaN
df = df.reindex(full_range)
df.index.name = "tanggal"

# isi NaN dengan metode LOCF (forward fill)
df = df.ffill()

print(f"Setelah LOCF: {len(df)} baris (tanggal lengkap)\n")

# ============================
# 3. Normalisasi Min-Max
# ============================

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index,
)

print("Min-Max Scaling diterapkan.")
print(f"Range setelah normalisasi: min={df_normalized.min().min():.2f}, max={df_normalized.max().max():.2f}")

# ============================
# 4. Simpan hasil
# ============================

# reset index agar tanggal jadi kolom kembali
df_normalized = df_normalized.reset_index().rename(columns={"index": "tanggal"})

# format tanggal ke YYYY-MM-DD
df_normalized["tanggal"] = df_normalized["tanggal"].dt.strftime("%Y-%m-%d")

output_path = os.path.join(script_dir, "dataset_preprocessed.csv")
df_normalized.to_csv(output_path, index=False)

print(f"\nHasil akhir: {df_normalized.shape[0]} baris x {df_normalized.shape[1]} kolom")
print(f"[OK] Dataset preprocessed disimpan ke: {output_path}")
print(f"\nPreview 5 baris pertama:")
print(df_normalized.head())
print(f"\nPreview 5 baris terakhir:")
print(df_normalized.tail())
