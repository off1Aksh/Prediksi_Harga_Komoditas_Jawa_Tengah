import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Lambda, Concatenate

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. LOAD DATA & REKONSTRUKSI SCALER
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
data_final_dir = os.path.join(script_dir, "..", "Dataset Final")

# --- Load data asli untuk rekonstruksi scaler ---
df_raw = pd.read_csv(os.path.join(data_final_dir, "dataset_harga_komoditas_2021_2026.csv"))
df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], format="%d/ %m/ %Y")

kolom_harga = [col for col in df_raw.columns if col != "tanggal"]
for col in kolom_harga:
    df_raw[col] = df_raw[col].astype(str).str.replace(",", "").astype(float)

df_raw = df_raw.set_index("tanggal")
full_range = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq="D")
df_raw = df_raw.reindex(full_range).ffill()

# fit scaler pada data asli (sama persis dengan pre-processing.py)
scaler = MinMaxScaler()
scaler.fit(df_raw.values)

# --- Load data preprocessed (sudah normalized) ---
df = pd.read_csv(os.path.join(data_final_dir, "dataset_preprocessed.csv"))
df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df.set_index("tanggal")

data = df.values  # shape: (1824, 8)
n_features = data.shape[1]

# ============================================================
# 1b. PEMETAAN GRUP FITUR BERDASARKAN VOLATILITAS
# ============================================================
# Urutan kolom di CSV (tanpa 'tanggal'):
#   0: bawang_merah, 1: bawang_putih, 2: beras,
#   3: cabai_merah,  4: cabai_rawit,  5: daging_ayam,
#   6: minyak_goreng, 7: telur_ayam

IDX_STABIL  = [1, 2, 6]           # bawang_putih, beras, minyak_goreng
IDX_VOLATIL = [0, 3, 4, 5, 7]     # bawang_merah, cabai_merah, cabai_rawit, daging_ayam, telur_ayam

# Bangun peta invers untuk reordering setelah concatenation
# Concatenated order: [stabil_0, stabil_1, stabil_2, volatil_0, ..., volatil_4]
# → posisi kolom asli di tensor concatenated
_concat_order = IDX_STABIL + IDX_VOLATIL          # [1,2,6,0,3,4,5,7]
REORDER_INDICES = [_concat_order.index(i) for i in range(n_features)]  # peta kembali ke urutan 0..7

print(f"Data loaded: {data.shape[0]} baris x {n_features} fitur")
print(f"Kolom: {kolom_harga}")
print(f"Grup Stabil  (idx {IDX_STABIL}): {[kolom_harga[i] for i in IDX_STABIL]}")
print(f"Grup Volatil (idx {IDX_VOLATIL}): {[kolom_harga[i] for i in IDX_VOLATIL]}")
print(f"Periode: {df.index.min().date()} - {df.index.max().date()}\n")


# ============================================================
# 2. FUNGSI SLIDING WINDOW
# ============================================================

def create_sequences(data, window_size):
    """
    Membuat pasangan X (input) dan Y (target) dari data time series.

    X[i] = data[i : i + window_size]        -> blok historis (window_size, 8)
    Y[i] = data[i + window_size]            -> target 1 hari ke depan (8,)
    """
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        Y.append(data[i + window_size])
    return np.array(X), np.array(Y)


# ============================================================
# 3. FUNGSI EVALUASI (RMSE, MAE, MAPE)
# ============================================================

def mean_absolute_percentage_error(y_true, y_pred):
    """Hitung MAPE, hindari pembagian nol."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred, kolom_harga):
    """
    Hitung RMSE, MAE, MAPE per komoditas.
    Input sudah dalam skala asli (Rupiah).
    """
    results = {}
    for i, col in enumerate(kolom_harga):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        mae = mean_absolute_error(true_col, pred_col)
        mape = mean_absolute_percentage_error(true_col, pred_col)

        results[col] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    return results


# ============================================================
# 3b. FUNGSI VISUALISASI
# ============================================================

def plot_loss_curve(history, nama):
    """Grafik training loss vs validation loss."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_title(f"{nama} - Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, kolom_harga, nama, n_points=100):
    """
    Grafik Prediksi vs Aktual untuk tiap komoditas.
    Menampilkan n_points terakhir agar grafik lebih jelas.
    """
    n_cols = 2
    n_rows = 4  # 8 komoditas / 2 kolom
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12))
    fig.suptitle(f"{nama} - Prediksi vs Aktual (Harga Asli Rp)",
                 fontsize=15, fontweight="bold", y=1.01)

    for i, col in enumerate(kolom_harga):
        row, c = divmod(i, n_cols)
        ax = axes[row][c]

        actual = y_true[-n_points:, i]
        predicted = y_pred[-n_points:, i]

        ax.plot(actual, label="Aktual", linewidth=1.5, alpha=0.9)
        ax.plot(predicted, label="Prediksi", linewidth=1.5, alpha=0.9, linestyle="--")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


def plot_summary_bar(summary_df):
    """Bar chart perbandingan Avg MAPE antar skenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    bars = ax.bar(summary_df["Skenario"], summary_df["Avg MAPE (%)"],
                  color=colors, edgecolor="black", linewidth=0.5)

    # highlight skenario terbaik
    best_idx = summary_df["Avg MAPE (%)"].idxmin()
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2.5)

    for bar, val in zip(bars, summary_df["Avg MAPE (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_title("Perbandingan Avg MAPE (%) per Skenario", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg MAPE (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. SKENARIO EKSPERIMEN
# ============================================================

skenario = [
    {"nama": "Skenario 1", "window_size": 7,  "lstm_units": 32},
    {"nama": "Skenario 2", "window_size": 7,  "lstm_units": 64},
    {"nama": "Skenario 3", "window_size": 14, "lstm_units": 32},
    {"nama": "Skenario 4", "window_size": 14, "lstm_units": 64},
    {"nama": "Skenario 5", "window_size": 30, "lstm_units": 32},
    {"nama": "Skenario 6", "window_size": 30, "lstm_units": 64},
]

EPOCHS = 50
BATCH_SIZE = 32

all_results = {}

for sk in skenario:
    nama = sk["nama"]
    window_size = sk["window_size"]
    lstm_units = sk["lstm_units"]

    print("=" * 70)
    print(f" {nama}: Window Size = {window_size}, LSTM Units = {lstm_units}")
    print("=" * 70)

    # --- Buat sequences ---
    X, Y = create_sequences(data, window_size)
    print(f"  Total sequences: {len(X)}")
    print(f"  X shape: {X.shape}  |  Y shape: {Y.shape}")

    # --- Chronological split 80/20 ---
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # --- Bangun model LSTM (Dual-Branch Functional API) ---
    inputs = Input(shape=(window_size, n_features), name="input_all")

    # Pecah input menjadi 2 cabang berdasarkan indeks volatilitas
    branch_stabil_input  = Lambda(
        lambda x: tf.gather(x, IDX_STABIL, axis=-1),
        name="slice_stabil"
    )(inputs)   # shape: (batch, window_size, 3)

    branch_volatil_input = Lambda(
        lambda x: tf.gather(x, IDX_VOLATIL, axis=-1),
        name="slice_volatil"
    )(inputs)   # shape: (batch, window_size, 5)

    # Cabang Stabil: LSTM 32 unit → Dense(3)
    x_stabil = LSTM(32, activation="tanh", name="lstm_stabil")(branch_stabil_input)
    out_stabil = Dense(len(IDX_STABIL), name="dense_stabil")(x_stabil)   # (batch, 3)

    # Cabang Volatil: LSTM 64 unit → Dense(5)
    x_volatil = LSTM(64, activation="tanh", name="lstm_volatil")(branch_volatil_input)
    out_volatil = Dense(len(IDX_VOLATIL), name="dense_volatil")(x_volatil)  # (batch, 5)

    # Gabung & reorder agar urutan kolom kembali seperti dataset asli
    merged = Concatenate(axis=-1, name="concat")([out_stabil, out_volatil])  # (batch, 8) — urutan: stabil lalu volatil
    final_output = Lambda(
        lambda x: tf.gather(x, REORDER_INDICES, axis=-1),
        name="reorder"
    )(merged)  # (batch, 8) — urutan kolom asli

    model = Model(inputs=inputs, outputs=final_output, name="dual_branch_lstm")
    model.compile(optimizer="adam", loss="mse")

    # --- Training ---
    print(f"  Training ({EPOCHS} epochs)...")
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        verbose=0,
    )

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    print(f"  Final Loss -> Train: {train_loss:.6f}  |  Val: {val_loss:.6f}")

    # --- Grafik Loss Curve ---
    plot_loss_curve(history, nama)

    # --- Prediksi pada data uji ---
    Y_pred_norm = model.predict(X_test, verbose=0)

    # --- Inverse transform ke skala asli (Rupiah) ---
    Y_test_real = scaler.inverse_transform(Y_test)
    Y_pred_real = scaler.inverse_transform(Y_pred_norm)

    # --- Evaluasi ---
    results = evaluate_model(Y_test_real, Y_pred_real, kolom_harga)
    all_results[nama] = results

    print(f"\n  {'Komoditas':<18} {'RMSE':>12} {'MAE':>12} {'MAPE (%)':>12}")
    print(f"  {'-'*54}")
    for col in kolom_harga:
        r = results[col]
        print(f"  {col:<18} {r['RMSE']:>12,.2f} {r['MAE']:>12,.2f} {r['MAPE']:>11.2f}%")
    print()

    # --- Grafik Prediksi vs Aktual ---
    plot_predictions(Y_test_real, Y_pred_real, kolom_harga, nama)

    # simpan model per skenario
    # model_path = os.path.join(data_final_dir, f"model_{nama.replace(' ', '_').lower()}.keras")
    # model.save(model_path)
    # print(f"  Model disimpan: {model_path}\n")


# ============================================================
# 5. RINGKASAN SEMUA SKENARIO
# ============================================================

print("\n" + "=" * 70)
print(" RINGKASAN SEMUA SKENARIO — Rata-rata Metrik (Semua Komoditas)")
print("=" * 70)

summary_rows = []
for nama, results in all_results.items():
    avg_rmse = np.mean([results[c]["RMSE"] for c in kolom_harga])
    avg_mae = np.mean([results[c]["MAE"] for c in kolom_harga])
    avg_mape = np.mean([results[c]["MAPE"] for c in kolom_harga])
    summary_rows.append({
        "Skenario": nama,
        "Avg RMSE": avg_rmse,
        "Avg MAE": avg_mae,
        "Avg MAPE (%)": avg_mape,
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# tentukan skenario terbaik berdasarkan MAPE terendah
best = summary_df.loc[summary_df["Avg MAPE (%)"].idxmin()]
print(f"\n>> Skenario terbaik: {best['Skenario']} "
      f"(Avg MAPE: {best['Avg MAPE (%)']:.2f}%)")

# --- Grafik Perbandingan Skenario ---
plot_summary_bar(summary_df)
