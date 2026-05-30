import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# KONFIGURASI GLOBAL
# ============================================================
EPOCHS = 1000
BATCH_SIZE = 32

script_dir = os.path.dirname(os.path.abspath(__file__))
hasil_dir = os.path.join(script_dir, "hasil_eksperimen_2")

os.makedirs(hasil_dir, exist_ok=True)

ORDERED_COLUMNS = [
    'bawang_merah', 'bawang_putih', 'beras', 'cabai_merah', 
    'cabai_rawit', 'daging_ayam', 'minyak_goreng', 'telur_ayam'
]

LABEL_MAP = {
    'bawang_merah': 'Bawang Merah', 'bawang_putih': 'Bawang Putih', 'beras': 'Beras',
    'cabai_merah': 'Cabai Merah', 'cabai_rawit': 'Cabai Rawit', 'daging_ayam': 'Daging Ayam',
    'minyak_goreng': 'Minyak Goreng', 'telur_ayam': 'Telur Ayam'
}

# 6 Skenario Eksperimen
configs = [
    {"skenario": 1, "window": 7,  "lstm": 32},
    {"skenario": 2, "window": 7,  "lstm": 64},
    {"skenario": 3, "window": 14, "lstm": 32},
    {"skenario": 4, "window": 14, "lstm": 64},
    {"skenario": 5, "window": 30, "lstm": 32},
    {"skenario": 6, "window": 30, "lstm": 64},
]

# ============================================================
# LOAD & PREPROCESS DATA
# ============================================================
print("Memuat dataset 2017-2026...")
df_raw = pd.read_csv(os.path.join(script_dir, "dataset_harga_komoditas_2017_2026.csv"))
df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], format="%d/ %m/ %Y", errors="coerce")
if df_raw["tanggal"].isnull().all():
    df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"])

for col in ORDERED_COLUMNS:
    df_raw[col] = df_raw[col].astype(str).str.replace(",", "").astype(float)

df_raw = df_raw.set_index("tanggal")
full_range = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq="D")
df_raw = df_raw.reindex(full_range).ffill()[ORDERED_COLUMNS]

# SCALER
scaler_master = MinMaxScaler()
scaler_master.fit(df_raw.values)
# Simpan scaler ke folder hasil eksperimen
joblib.dump(scaler_master, os.path.join(hasil_dir, "scaler_master.gz"))

scaled_data = scaler_master.transform(df_raw.values)

# Fungsi Helper
def create_sequences(data_col, window_size):
    X, Y = [], []
    for i in range(len(data_col) - window_size):
        X.append(data_col[i:i+window_size])
        Y.append(data_col[i+window_size])
    return np.array(X), np.array(Y)

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ============================================================
# EKSEKUSI EKSPERIMEN
# ============================================================
rekap_akhir = []

for cfg in configs:
    s_idx = cfg['skenario']
    w_size = cfg['window']
    l_units = cfg['lstm']
    
    skenario_name = f"Skenario_{s_idx}_W{w_size}_L{l_units}"
    s_dir = os.path.join(hasil_dir, skenario_name)
    s_model_dir = os.path.join(s_dir, "models")
    os.makedirs(s_model_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"🚀 MULAI EKSEKUSI: {skenario_name}")
    print("=" * 60)
    
    evaluasi_rows = []
    
    # Untuk Plotting
    n_points = 100
    fig_pred, axes_pred = plt.subplots(4, 2, figsize=(16, 16))
    fig_pred.suptitle(f"Prediksi vs Aktual ({skenario_name})\n100 Hari Terakhir Data Testing", fontsize=16, fontweight='bold')
    
    fig_loss, axes_loss = plt.subplots(4, 2, figsize=(16, 16))
    fig_loss.suptitle(f"Kurva Training Loss vs Val_Loss ({skenario_name})", fontsize=16, fontweight='bold')
    
    # EarlyStopping dengan perlindungan jebakan awal (start_from_epoch=50) dan toleransi (patience=50)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0, start_from_epoch=50)
    
    for c_idx, col_name in enumerate(ORDERED_COLUMNS):
        print(f"  > Sedang melatih {LABEL_MAP[col_name]}...")
        
        # Data Preparation
        col_data = scaled_data[:, c_idx].reshape(-1, 1)
        X, Y = create_sequences(col_data, w_size)
        
        # 80/20 Split secara kronologis
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]
        
        # Build Model
        model = Sequential([
            LSTM(l_units, activation='tanh', input_shape=(w_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train Model
        hist = model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Ambil epoch yang menghasilkan val_loss terendah
        best_ep = early_stop.best_epoch + 1
        
        # Prediksi & Evaluasi
        Y_pred_norm = model.predict(X_test, verbose=0).flatten()
        
        # Kembalikan ke nilai asli (Inverse Transform)
        dummy_test = np.zeros((len(Y_test), 8))
        dummy_pred = np.zeros((len(Y_pred_norm), 8))
        dummy_test[:, c_idx] = Y_test.flatten()
        dummy_pred[:, c_idx] = Y_pred_norm
        
        Y_test_real = scaler_master.inverse_transform(dummy_test)[:, c_idx]
        Y_pred_real = scaler_master.inverse_transform(dummy_pred)[:, c_idx]
        
        rmse = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real))
        mae = mean_absolute_error(Y_test_real, Y_pred_real)
        mape = mean_absolute_percentage_error(Y_test_real, Y_pred_real)
        
        evaluasi_rows.append({
            'Komoditas': LABEL_MAP[col_name],
            'RMSE (Rp)': round(rmse, 2),
            'MAE (Rp)': round(mae, 2),
            'MAPE (%)': round(mape, 2),
            'Epoch Berhenti': best_ep
        })
        
        # Save Model Sementara
        model.save(os.path.join(s_model_dir, f"model_{col_name}.keras"))
        
        # -- Plotting Prediksi --
        row, c = divmod(c_idx, 2)
        ax_p = axes_pred[row][c]
        ax_p.plot(Y_test_real[-n_points:], label='Aktual', linewidth=1.5, color='#2c3e50')
        ax_p.plot(Y_pred_real[-n_points:], label='Prediksi Univariate', linewidth=1.2, color='#e74c3c', linestyle='--')
        ax_p.set_title(LABEL_MAP[col_name], fontsize=12, fontweight='bold')
        ax_p.legend(fontsize=8)
        ax_p.grid(True, alpha=0.3)
        
        # -- Plotting Loss --
        ax_l = axes_loss[row][c]
        ax_l.plot(hist.history['loss'], label='Train Loss', color='#3498db')
        ax_l.plot(hist.history['val_loss'], label='Val Loss', color='#e67e22')
        ax_l.axvline(x=best_ep-1, color='red', linestyle=':', label='Best Epoch')
        ax_l.set_title(LABEL_MAP[col_name], fontsize=12, fontweight='bold')
        ax_l.legend(fontsize=8)
        ax_l.grid(True, alpha=0.3)

    # Simpan Hasil Evaluasi per Skenario
    df_eval = pd.DataFrame(evaluasi_rows)
    df_eval.to_csv(os.path.join(s_dir, f"evaluasi_{skenario_name}.csv"), index=False)
    
    # Simpan Gambar
    fig_pred.tight_layout()
    fig_pred.savefig(os.path.join(s_dir, "grafik_prediksi_vs_aktual.png"), dpi=150)
    plt.close(fig_pred)
    
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join(s_dir, "grafik_loss_history.png"), dpi=150)
    plt.close(fig_loss)
    
    # Kalkulasi Rata-rata Skenario ini
    avg_rmse = df_eval['RMSE (Rp)'].mean()
    avg_mape = df_eval['MAPE (%)'].mean()
    
    rekap_akhir.append({
        'Skenario': s_idx,
        'Window Size': w_size,
        'LSTM Units': l_units,
        'Rata-rata RMSE': round(avg_rmse, 2),
        'Rata-rata MAPE (%)': round(avg_mape, 2)
    })
    
    print(f"✅ [SELESAI] {skenario_name}")
    print(f"   Rata-rata MAPE: {avg_mape:.2f}% | Rata-rata RMSE: Rp{avg_rmse:.2f}")

# ============================================================
# PENENTUAN JUARA & EXPORT MODEL
# ============================================================
df_rekap = pd.DataFrame(rekap_akhir)
df_rekap.to_csv(os.path.join(hasil_dir, "rekapitulasi_final.csv"), index=False)

# Tentukan juara berdasarkan MAPE terendah
best_scenario_row = df_rekap.loc[df_rekap['Rata-rata MAPE (%)'].idxmin()]
best_s_idx = int(best_scenario_row['Skenario'])
best_w = int(best_scenario_row['Window Size'])
best_l = int(best_scenario_row['LSTM Units'])

best_name = f"Skenario_{best_s_idx}_W{best_w}_L{best_l}"

print("\n" + "=" * 60)
print(f"🏆 EKSPERIMEN SKRIPSI SELESAI! 🏆")
print("=" * 60)
print(df_rekap.to_string(index=False))
print("-" * 60)
print(f"🏅 SKENARIO TERBAIK: {best_name}")
print(f"Model terbaik tersimpan di: {os.path.join(hasil_dir, best_name, 'models')}")
print("Silakan copy/pindahkan model tersebut secara manual ke folder 'backend/models_univariate' jika Anda sudah puas dengan hasilnya.")
