"""
ANALISIS PERBANDINGAN: UNIVARIATE vs MULTIVARIATE DUAL-BRANCH LSTM
====================================================================
Arsitektur dari coba1.py:
  - Dual-Branch (Stabil vs Volatil)
  - Stabil  (LSTM 32, tanh): bawang_putih, beras, minyak_goreng
  - Volatil (LSTM 64, tanh): bawang_merah, cabai_merah, cabai_rawit, daging_ayam, telur_ayam

Perbandingan:
  - UNIVARIATE:   8 model LSTM terpisah, masing-masing 1 fitur
  - MULTIVARIATE: 1 model Dual-Branch Functional API, 8 fitur sekaligus

Konfigurasi training mengikuti skenario terbaik dari coba1.py.
Hasil disimpan ke folder: hasil_analisis_dual_branch/

TIDAK MENGUBAH KODE APAPUN YANG SUDAH ADA.
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
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# KONFIGURASI
# ============================================================
EPOCHS = 50
BATCH_SIZE = 32
WINDOW_SIZE = 30  # Skenario terbaik dari coba1.py (Skenario 4)
LSTM_UNITS = 64    # Skenario terbaik dari coba1.py (Skenario 4)

# Grup berdasarkan volatilitas (sama seperti coba1.py)
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
# LOAD DATA
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, "..")
data_final_dir = os.path.join(backend_dir, "Dataset Final")
output_dir = os.path.join(backend_dir, "hasil_analisis_dual_branch_30")
os.makedirs(output_dir, exist_ok=True)

# Load data asli untuk rekonstruksi scaler
df_raw = pd.read_csv(os.path.join(data_final_dir, "dataset_harga_komoditas_2021_2026.csv"))
df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], format="%d/ %m/ %Y")

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
df = pd.read_csv(os.path.join(data_final_dir, "dataset_preprocessed.csv"))
df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df.set_index("tanggal")

data = df.values
n_features = data.shape[1]

# Reorder indices
_concat_order = IDX_STABIL + IDX_VOLATIL
REORDER_INDICES = [_concat_order.index(i) for i in range(n_features)]

print(f"Dataset: {data.shape[0]} baris x {n_features} fitur")
print(f"Periode: {df.index.min().date()} -- {df.index.max().date()}")
print(f"Window Size: {WINDOW_SIZE}, LSTM Units: {LSTM_UNITS}, Epochs: {EPOCHS}")
print(f"Grup Stabil:  {[kolom_harga[i] for i in IDX_STABIL]}")
print(f"Grup Volatil: {[kolom_harga[i] for i in IDX_VOLATIL]}")
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
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print()


# ============================================================
# 1. TRAIN MULTIVARIATE DUAL-BRANCH (arsitektur coba1.py)
# ============================================================
print("=" * 70)
print(" TRAINING: MULTIVARIATE DUAL-BRANCH LSTM")
print(f" Arsitektur: Stabil(LSTM 32) + Volatil(LSTM 64) -- Functional API")
print("=" * 70)

inputs = Input(shape=(WINDOW_SIZE, n_features), name="input_all")

branch_stabil_input = Lambda(
    lambda x: tf.gather(x, IDX_STABIL, axis=-1),
    name="slice_stabil"
)(inputs)

branch_volatil_input = Lambda(
    lambda x: tf.gather(x, IDX_VOLATIL, axis=-1),
    name="slice_volatil"
)(inputs)

# Cabang Stabil: LSTM 32 unit (sama seperti coba1.py)
x_stabil = LSTM(32, activation="tanh", name="lstm_stabil")(branch_stabil_input)
out_stabil = Dense(len(IDX_STABIL), name="dense_stabil")(x_stabil)

# Cabang Volatil: LSTM 64 unit (sama seperti coba1.py)
x_volatil = LSTM(64, activation="tanh", name="lstm_volatil")(branch_volatil_input)
out_volatil = Dense(len(IDX_VOLATIL), name="dense_volatil")(x_volatil)

merged = Concatenate(axis=-1, name="concat")([out_stabil, out_volatil])
final_output = Lambda(
    lambda x: tf.gather(x, REORDER_INDICES, axis=-1),
    name="reorder"
)(merged)

model_multi = Model(inputs=inputs, outputs=final_output, name="dual_branch_lstm")
model_multi.compile(optimizer="adam", loss="mse")

print("Training Dual-Branch...")
history_multi = model_multi.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, Y_test),
    verbose=0,
)

train_loss = history_multi.history["loss"][-1]
val_loss = history_multi.history["val_loss"][-1]
print(f"Final Loss -> Train: {train_loss:.6f} | Val: {val_loss:.6f}")

# Prediksi & inverse transform
Y_pred_multi_norm = model_multi.predict(X_test, verbose=0)
Y_test_real = scaler.inverse_transform(Y_test)
Y_pred_multi_real = scaler.inverse_transform(Y_pred_multi_norm)

print("Multivariate Dual-Branch selesai!\n")


# ============================================================
# 2. TRAIN UNIVARIATE (8 model terpisah)
# ============================================================
print("=" * 70)
print(" TRAINING: 8x UNIVARIATE LSTM (satu model per komoditas)")
print("=" * 70)

Y_pred_uni_real = np.zeros_like(Y_pred_multi_real)

for idx, col in enumerate(kolom_harga):
    print(f"  Training Univariate: {LABEL_MAP[col]}...")
    
    # Ambil data 1 kolom
    col_data = data[:, idx].reshape(-1, 1)
    
    # Buat sequences per komoditas
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
    model_uni.fit(X_u_train, Y_u_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_u_test, Y_u_test), verbose=0)
    
    # Prediksi & inverse transform per komoditas
    pred_u_norm = model_uni.predict(X_u_test, verbose=0).flatten()
    
    # Inverse transform: buat dummy array penuh, isi kolom ini saja
    dummy_full = np.zeros((len(pred_u_norm), n_features))
    dummy_full[:, idx] = pred_u_norm
    pred_u_real = scaler.inverse_transform(dummy_full)[:, idx]
    
    Y_pred_uni_real[:, idx] = pred_u_real

print("Semua 8 model Univariate selesai!\n")


# ============================================================
# 3. PERBANDINGAN RMSE, MAE, MAPE
# ============================================================
print("=" * 70)
print(" PERBANDINGAN: UNIVARIATE vs MULTIVARIATE DUAL-BRANCH")
print("=" * 70)

header = f"  {'Komoditas':<18} {'RMSE Uni':>12} {'RMSE Multi':>12} {'MAE Uni':>10} {'MAE Multi':>10} {'MAPE Uni':>10} {'MAPE Multi':>10} {'Pemenang RMSE':>14}"
print(header)
print("  " + "-" * 100)

comparison_rows = []
multi_wins_rmse = 0
multi_wins_mape = 0

for idx, col in enumerate(kolom_harga):
    rmse_uni = np.sqrt(mean_squared_error(Y_test_real[:, idx], Y_pred_uni_real[:, idx]))
    rmse_multi = np.sqrt(mean_squared_error(Y_test_real[:, idx], Y_pred_multi_real[:, idx]))
    mae_uni = mean_absolute_error(Y_test_real[:, idx], Y_pred_uni_real[:, idx])
    mae_multi = mean_absolute_error(Y_test_real[:, idx], Y_pred_multi_real[:, idx])
    mape_uni = mean_absolute_percentage_error(Y_test_real[:, idx], Y_pred_uni_real[:, idx])
    mape_multi = mean_absolute_percentage_error(Y_test_real[:, idx], Y_pred_multi_real[:, idx])
    
    winner_rmse = "MULTI" if rmse_multi < rmse_uni else "UNI"
    if rmse_multi < rmse_uni:
        multi_wins_rmse += 1
    if mape_multi < mape_uni:
        multi_wins_mape += 1
    
    comparison_rows.append({
        'Komoditas': LABEL_MAP[col],
        'Grup': 'Stabil' if idx in IDX_STABIL else 'Volatil',
        'RMSE_Univariate': round(rmse_uni, 2),
        'RMSE_Multivariate': round(rmse_multi, 2),
        'MAE_Univariate': round(mae_uni, 2),
        'MAE_Multivariate': round(mae_multi, 2),
        'MAPE_Univariate': round(mape_uni, 2),
        'MAPE_Multivariate': round(mape_multi, 2),
        'Pemenang_RMSE': 'Multivariate' if rmse_multi < rmse_uni else 'Univariate',
        'Pemenang_MAPE': 'Multivariate' if mape_multi < mape_uni else 'Univariate'
    })
    
    print(f"  {LABEL_MAP[col]:<18} {rmse_uni:>10,.2f}Rp {rmse_multi:>10,.2f}Rp {mae_uni:>9,.2f}Rp {mae_multi:>9,.2f}Rp {mape_uni:>8.2f}% {mape_multi:>9.2f}% {'  ' + winner_rmse:>14}")

print("  " + "-" * 100)
print(f"\n  Multivariate MENANG RMSE: {multi_wins_rmse}/8 komoditas")
print(f"  Multivariate MENANG MAPE: {multi_wins_mape}/8 komoditas")

# Simpan ke CSV
comp_df = pd.DataFrame(comparison_rows)
csv_path = os.path.join(output_dir, "perbandingan_univariate_vs_dual_branch.csv")
comp_df.to_csv(csv_path, index=False)
print(f"\n[SAVED] {csv_path}")


# ============================================================
# 4. VISUALISASI: BAR CHART PERBANDINGAN RMSE
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(kolom_harga))
width = 0.35

bars_uni = ax.bar(x - width/2, [r['RMSE_Univariate'] for r in comparison_rows], width,
                   label='Univariate (1 fitur per model)', color='#e74c3c', alpha=0.85)
bars_multi = ax.bar(x + width/2, [r['RMSE_Multivariate'] for r in comparison_rows], width,
                     label='Multivariate Dual-Branch (8 fitur)', color='#2ecc71', alpha=0.85)

ax.set_xlabel('Komoditas', fontsize=12)
ax.set_ylabel('RMSE (Rupiah) - Lebih rendah = lebih baik', fontsize=11)
ax.set_title(f'Univariate vs Multivariate DUAL-BRANCH LSTM\n(Window={WINDOW_SIZE}, Stabil LSTM=32, Volatil LSTM=64, Epochs={EPOCHS})',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABEL_MAP[c] for c in kolom_harga], rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars_uni:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=7, color='#c0392b')
for bar in bars_multi:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=7, color='#27ae60')

plt.tight_layout()
rmse_path = os.path.join(output_dir, "perbandingan_rmse.png")
plt.savefig(rmse_path, dpi=200)
plt.close()
print(f"[SAVED] {rmse_path}")


# ============================================================
# 5. VISUALISASI: BAR CHART PERBANDINGAN MAPE
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(kolom_harga))

bars_uni = ax.bar(x - width/2, [r['MAPE_Univariate'] for r in comparison_rows], width,
                   label='Univariate (1 fitur per model)', color='#e74c3c', alpha=0.85)
bars_multi = ax.bar(x + width/2, [r['MAPE_Multivariate'] for r in comparison_rows], width,
                     label='Multivariate Dual-Branch (8 fitur)', color='#2ecc71', alpha=0.85)

ax.set_xlabel('Komoditas', fontsize=12)
ax.set_ylabel('MAPE (%) - Lebih rendah = lebih baik', fontsize=11)
ax.set_title(f'Univariate vs Multivariate DUAL-BRANCH LSTM (MAPE)\n(Window={WINDOW_SIZE}, Stabil LSTM=32, Volatil LSTM=64, Epochs={EPOCHS})',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABEL_MAP[c] for c in kolom_harga], rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars_uni:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=7, color='#c0392b')
for bar in bars_multi:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=7, color='#27ae60')

plt.tight_layout()
mape_path = os.path.join(output_dir, "perbandingan_mape.png")
plt.savefig(mape_path, dpi=200)
plt.close()
print(f"[SAVED] {mape_path}")


# ============================================================
# 6. VISUALISASI: PREDIKSI vs AKTUAL (per komoditas)
# ============================================================
n_points = 100
fig, axes = plt.subplots(4, 2, figsize=(16, 16))
fig.suptitle(f"Dual-Branch LSTM: Prediksi vs Aktual ({n_points} titik terakhir)\n"
             f"Window={WINDOW_SIZE}, Stabil LSTM=32, Volatil LSTM=64",
             fontsize=14, fontweight='bold')

for i, col in enumerate(kolom_harga):
    row, c = divmod(i, 2)
    ax = axes[row][c]
    
    actual = Y_test_real[-n_points:, i]
    pred_multi = Y_pred_multi_real[-n_points:, i]
    pred_uni = Y_pred_uni_real[-n_points:, i]
    
    ax.plot(actual, label='Aktual', linewidth=1.5, color='#2c3e50', alpha=0.9)
    ax.plot(pred_multi, label='Multivariate DB', linewidth=1.2, color='#2ecc71', alpha=0.8, linestyle='--')
    ax.plot(pred_uni, label='Univariate', linewidth=1.2, color='#e74c3c', alpha=0.8, linestyle=':')
    
    grup = 'Stabil' if i in IDX_STABIL else 'Volatil'
    ax.set_title(f"{LABEL_MAP[col]} [{grup}]", fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

plt.tight_layout()
pred_path = os.path.join(output_dir, "prediksi_vs_aktual.png")
plt.savefig(pred_path, dpi=200)
plt.close()
print(f"[SAVED] {pred_path}")


# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 70)
print(" RINGKASAN: UNIVARIATE vs DUAL-BRANCH LSTM")
print("=" * 70)
print(f"""
  Konfigurasi:
    - Window Size: {WINDOW_SIZE}
    - LSTM Units: Stabil=32, Volatil=64
    - Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}
    - Split: 80% train, 20% test

  Hasil:
    - Multivariate menang RMSE: {multi_wins_rmse}/8 komoditas
    - Multivariate menang MAPE: {multi_wins_mape}/8 komoditas

  File tersimpan di: {output_dir}
    - perbandingan_univariate_vs_dual_branch.csv
    - perbandingan_rmse.png
    - perbandingan_mape.png
    - prediksi_vs_aktual.png
""")
