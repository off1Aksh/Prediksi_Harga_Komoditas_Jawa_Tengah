"""
ANALISIS PERBANDINGAN (DATASET 2017-2026): UNIVARIATE vs MULTIVARIATE DUAL-BRANCH LSTM
======================================================================================
Menggunakan Early Stopping dengan batas 1500 Epochs.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Lambda, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# KONFIGURASI
# ============================================================
EPOCHS = 1500      # Batas maksimal (digunakan dengan EarlyStopping)
BATCH_SIZE = 32
WINDOW_SIZE = 30

# Grup berdasarkan volatilitas
IDX_STABIL  = [1, 2, 6]           # bawang_putih, beras, minyak_goreng
IDX_VOLATIL = [0, 3, 4, 5, 7]     # bawang_merah, cabai_merah, cabai_rawit, daging_ayam, telur_ayam

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
# LOAD DATA (Dataset 2017-2026)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "hasil_dual_branch_early_stopping")
os.makedirs(output_dir, exist_ok=True)

# Load data asli untuk rekonstruksi scaler
df_raw = pd.read_csv(os.path.join(script_dir, "dataset_harga_komoditas_2017_2026.csv"))
df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], format="%d/ %m/ %Y", errors="coerce")
if df_raw["tanggal"].isnull().all():
    df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"])

kolom_harga = [col for col in df_raw.columns if col != "tanggal"]
for col in kolom_harga:
    df_raw[col] = df_raw[col].astype(str).str.replace(",", "").astype(float)

df_raw = df_raw.set_index("tanggal")
full_range = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq="D")
df_raw = df_raw.reindex(full_range).ffill()

# Fit scaler pada data asli
scaler = MinMaxScaler()
scaler.fit(df_raw.values)

# Load data preprocessed
df = pd.read_csv(os.path.join(script_dir, "dataset_preprocessed.csv"))
df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df.set_index("tanggal")

data = df.values
n_features = data.shape[1]

# Reorder indices
_concat_order = IDX_STABIL + IDX_VOLATIL
REORDER_INDICES = [_concat_order.index(i) for i in range(n_features)]

print(f"Dataset: {data.shape[0]} baris x {n_features} fitur")
print(f"Window Size: {WINDOW_SIZE}, Max Epochs: {EPOCHS}")
print()

# ============================================================
# FUNGSI HELPER
# ============================================================
def create_sequences(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
    return np.array(X), np.array(Y)

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ============================================================
# BUAT SEQUENCES & SPLIT
# ============================================================
X, Y = create_sequences(data, WINDOW_SIZE)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

print(f"Total sequences: {len(X)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# EARLY STOPPING
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# ============================================================
# 1. TRAIN MULTIVARIATE DUAL-BRANCH
# ============================================================
print("=" * 70)
print(" TRAINING: MULTIVARIATE DUAL-BRANCH LSTM (Stabil vs Volatil)")
print("=" * 70)

inputs = Input(shape=(WINDOW_SIZE, n_features), name="input_all")

branch_stabil_input = Lambda(lambda x: tf.gather(x, IDX_STABIL, axis=-1), name="slice_stabil")(inputs)
branch_volatil_input = Lambda(lambda x: tf.gather(x, IDX_VOLATIL, axis=-1), name="slice_volatil")(inputs)

x_stabil = LSTM(32, activation="tanh", name="lstm_stabil")(branch_stabil_input)
out_stabil = Dense(len(IDX_STABIL), name="dense_stabil")(x_stabil)

x_volatil = LSTM(64, activation="tanh", name="lstm_volatil")(branch_volatil_input)
out_volatil = Dense(len(IDX_VOLATIL), name="dense_volatil")(x_volatil)

merged = Concatenate(axis=-1, name="concat")([out_stabil, out_volatil])
final_output = Lambda(lambda x: tf.gather(x, REORDER_INDICES, axis=-1), name="reorder")(merged)

model_multi = Model(inputs=inputs, outputs=final_output, name="dual_branch_lstm")
model_multi.compile(optimizer="adam", loss="mse")

print("Training Dual-Branch dengan EarlyStopping...")
history_multi = model_multi.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop],
    verbose=0,
)

train_loss = history_multi.history["loss"][-1]
val_loss = history_multi.history["val_loss"][-1]
best_epoch_multi = len(history_multi.history['loss'])
print(f"Dual-Branch selesai di Epoch {best_epoch_multi} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n")

# Prediksi & inverse transform
Y_pred_multi_norm = model_multi.predict(X_test, verbose=0)
Y_test_real = scaler.inverse_transform(Y_test)
Y_pred_multi_real = scaler.inverse_transform(Y_pred_multi_norm)

# ============================================================
# 2. TRAIN UNIVARIATE (8 model terpisah)
# ============================================================
print("=" * 70)
print(" TRAINING: 8x UNIVARIATE LSTM (satu model per komoditas)")
print("=" * 70)

Y_pred_uni_real = np.zeros_like(Y_pred_multi_real)
best_epochs_uni = {}

for idx, col in enumerate(kolom_harga):
    print(f"  Training Univariate: {LABEL_MAP[col]}...")
    col_data = data[:, idx].reshape(-1, 1)
    
    X_u, Y_u = [], []
    for i in range(len(col_data) - WINDOW_SIZE):
        X_u.append(col_data[i:i+WINDOW_SIZE])
        Y_u.append(col_data[i+WINDOW_SIZE])
    X_u, Y_u = np.array(X_u), np.array(Y_u)
    
    X_u_train, X_u_test = X_u[:split_idx], X_u[split_idx:]
    Y_u_train, Y_u_test = Y_u[:split_idx], Y_u[split_idx:]
    
    model_uni = Sequential([
        LSTM(32, activation='tanh', input_shape=(WINDOW_SIZE, 1)),
        Dense(1)
    ])
    model_uni.compile(optimizer='adam', loss='mse')
    
    hist_u = model_uni.fit(
        X_u_train, Y_u_train, 
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_u_test, Y_u_test), 
        callbacks=[early_stop],
        verbose=0
    )
    
    # Cek apakah best epoch adalah epoch 1 (index 0)
    best_ep = early_stop.best_epoch + 1 if early_stop.stopped_epoch > 0 else len(hist_u.history['loss'])
    
    if early_stop.best_epoch == 0:
        print(f"    [Info] Early stopping mengembalikan Epoch 1. Memaksa ulang training ke 50 Epochs...")
        # Reset model dari awal
        model_uni = Sequential([
            LSTM(32, activation='tanh', input_shape=(WINDOW_SIZE, 1)),
            Dense(1)
        ])
        model_uni.compile(optimizer='adam', loss='mse')
        # Jalankan persis 50 epoch tanpa early stopping
        model_uni.fit(
            X_u_train, Y_u_train, 
            epochs=50, batch_size=BATCH_SIZE,
            validation_data=(X_u_test, Y_u_test), 
            verbose=0
        )
        best_ep = 50
    
    best_epochs_uni[LABEL_MAP[col]] = best_ep
    print(f"    Selesai di Epoch {best_ep}")
    
    pred_u_norm = model_uni.predict(X_u_test, verbose=0).flatten()
    
    dummy_full = np.zeros((len(pred_u_norm), n_features))
    dummy_full[:, idx] = pred_u_norm
    pred_u_real = scaler.inverse_transform(dummy_full)[:, idx]
    Y_pred_uni_real[:, idx] = pred_u_real

print("\nSemua 8 model Univariate selesai!\n")

# ============================================================
# 3. PERBANDINGAN RMSE & VISUALISASI
# ============================================================
comparison_rows = []
for idx, col in enumerate(kolom_harga):
    rmse_uni = np.sqrt(mean_squared_error(Y_test_real[:, idx], Y_pred_uni_real[:, idx]))
    rmse_multi = np.sqrt(mean_squared_error(Y_test_real[:, idx], Y_pred_multi_real[:, idx]))
    comparison_rows.append({
        'Komoditas': LABEL_MAP[col],
        'RMSE_Univariate': round(rmse_uni, 2),
        'RMSE_Multivariate': round(rmse_multi, 2),
    })

comp_df = pd.DataFrame(comparison_rows)
csv_path = os.path.join(output_dir, "perbandingan_univariate_vs_dual_branch.csv")
comp_df.to_csv(csv_path, index=False)
print(f"\n[SAVED] {csv_path}")

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(kolom_harga))
width = 0.35

bars_uni = ax.bar(x - width/2, [r['RMSE_Univariate'] for r in comparison_rows], width, color='#e74c3c')
bars_multi = ax.bar(x + width/2, [r['RMSE_Multivariate'] for r in comparison_rows], width, color='#2ecc71')

ax.set_title(f'Univariate vs Multivariate DUAL-BRANCH LSTM\n(Dataset 2017-2026, Window={WINDOW_SIZE}, EarlyStopping)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABEL_MAP[c] for c in kolom_harga], rotation=30)

plt.tight_layout()
rmse_path = os.path.join(output_dir, "perbandingan_rmse.png")
plt.savefig(rmse_path, dpi=200)
plt.close()
print(f"[SAVED] {rmse_path}")
