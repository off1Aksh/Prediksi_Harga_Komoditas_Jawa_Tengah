"""
TRAINING FINAL: 8 MODEL UNIVARIATE LSTM
=======================================
Script ini ditujukan untuk melatih model final untuk aplikasi (backend).
Data: Dataset 2017-2026
Konfigurasi: Univariate, Window=30, LSTM=32, Epochs=1500 (EarlyStopping start_from_epoch=100)
Output: 8 file model (.keras) dan 1 scaler (scaler_master.gz)
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# KONFIGURASI & PATH
# ============================================================
EPOCHS = 1500
BATCH_SIZE = 32

# Hasil dari Sequential Holdout Tuning (216 Kombinasi)
BEST_PARAMS = {
    'bawang_merah': {'window': 30, 'lstm': 64},
    'bawang_putih': {'window': 7, 'lstm': 128},
    'beras': {'window': 14, 'lstm': 128},
    'cabai_merah': {'window': 30, 'lstm': 128},
    'cabai_rawit': {'window': 30, 'lstm': 32},
    'daging_ayam': {'window': 30, 'lstm': 64},
    'minyak_goreng': {'window': 30, 'lstm': 128},
    'telur_ayam': {'window': 30, 'lstm': 64}
}

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, "..")
models_dir = os.path.join(backend_dir, "models_univariate")

os.makedirs(models_dir, exist_ok=True)

# Nama komoditas yang akan jadi nama file model (sesuai urutan di kolom)
ORDERED_COLUMNS = [
    'bawang_merah', 'bawang_putih', 'beras', 'cabai_merah', 
    'cabai_rawit', 'daging_ayam', 'minyak_goreng', 'telur_ayam'
]

LABEL_MAP = {
    'bawang_merah': 'Bawang Merah',
    'bawang_putih': 'Bawang Putih',
    'beras': 'Beras',
    'cabai_merah': 'Cabai Merah',
    'cabai_rawit': 'Cabai Rawit',
    'daging_ayam': 'Daging Ayam',
    'minyak_goreng': 'Minyak Goreng',
    'telur_ayam': 'Telur Ayam'
}

# ============================================================
# LOAD & PREPROCESS DATA
# ============================================================
print("1. Memuat dataset...")
# Load dataset mentah (Rupiah asli)
raw_path = os.path.join(script_dir, "dataset_harga_komoditas_2017_2026.csv")
df_raw = pd.read_csv(raw_path)

# Konversi tanggal dengan aman (seperti di app.py)
parsed_dates = pd.to_datetime(df_raw["tanggal"], format="%d/ %m/ %Y", errors="coerce")
if parsed_dates.isnull().all():
    df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], errors="coerce")
else:
    df_raw["tanggal"] = parsed_dates

kolom_harga = [col for col in df_raw.columns if col != "tanggal"]
for col in kolom_harga:
    df_raw[col] = df_raw[col].astype(str).str.replace(",", "").astype(float)

df_raw = df_raw.set_index("tanggal")
full_range = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq="D")
df_raw = df_raw.reindex(full_range).ffill()

# PASTIKAN URUTAN KOLOM SAMA SEPERTI ORDERED_COLUMNS
df_raw = df_raw[ORDERED_COLUMNS]

print(f"Total data: {df_raw.shape[0]} baris.")

# ============================================================
# PEMBUATAN SCALER GLOBAL
# ============================================================
print("\n2. Membuat Scaler Master...")
scaler_master = MinMaxScaler()
scaler_master.fit(df_raw.values)

# Simpan Scaler Master agar bisa dipakai oleh app.py
scaler_path = os.path.join(models_dir, "scaler_master.gz")
joblib.dump(scaler_master, scaler_path)
print(f"[SAVED] {scaler_path}")

# Karena kita training dengan seluruh data (100% data untuk model final), kita transform semua data
scaled_data = scaler_master.transform(df_raw.values)

# ============================================================
# FUNGSI HELPER SEQUENCES
# ============================================================
def create_sequences(data_col, window_size):
    X, Y = [], []
    for i in range(len(data_col) - window_size):
        X.append(data_col[i:i+window_size])
        Y.append(data_col[i+window_size])
    return np.array(X), np.array(Y)


# ============================================================
# TRAINING 8 MODEL UNIVARIATE (FULL DATASET)
# ============================================================
print("\n3. Memulai proses training 8 Model Univariate...")
print(f"Max Epochs={EPOCHS}, Batch Size={BATCH_SIZE}")
print("Catatan: Training ini menggunakan 100% data (tanpa validation split) untuk akurasi maksimal di production.")
print("-" * 50)

# Karena kita ingin model belajar dari SEMUA data sebelum di-deploy, kita tidak pakai validation split.
# Early stopping pada "val_loss" diganti menjadi "loss" karena tidak ada validasi.
# Ingat: Start from epoch 100.
early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=1, start_from_epoch=100)

for idx, col_name in enumerate(ORDERED_COLUMNS):
    # Ambil parameter khusus untuk komoditas ini
    window_size = BEST_PARAMS[col_name]['window']
    lstm_units = BEST_PARAMS[col_name]['lstm']
    
    print(f"\n> Training Model [{idx+1}/8]: {LABEL_MAP[col_name]}")
    print(f"  Konfigurasi Kustom -> Window: {window_size}, LSTM: {lstm_units}")
    
    # Ambil kolom spesifik dari data yang sudah di-scale
    col_data = scaled_data[:, idx].reshape(-1, 1)
    
    # Buat sekuens
    X_train, Y_train = create_sequences(col_data, window_size)
    
    # Bangun Arsitektur Univariate Spesifik
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Lakukan fit pada 100% data
    hist = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1 # Tampilkan progress bar karena agak lama
    )
    
    best_ep = early_stop.best_epoch + 1 if early_stop.stopped_epoch > 0 else len(hist.history['loss'])
    print(f"[Info] {LABEL_MAP[col_name]} selesai di Epoch {best_ep}")
    
    # Simpan model
    model_filename = f"model_{col_name}.keras"
    model_filepath = os.path.join(models_dir, model_filename)
    model.save(model_filepath)
    print(f"[SAVED] {model_filepath}")

print("\n" + "=" * 50)
print("SELURUH PROSES TRAINING SELESAI!")
print(f"8 Model dan 1 Scaler Master berhasil disimpan di direktori:")
print(f" -> {models_dir}")
print("=" * 50)
