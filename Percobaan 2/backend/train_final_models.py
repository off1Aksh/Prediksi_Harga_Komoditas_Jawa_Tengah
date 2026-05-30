import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Mematikan peringatan GPU yang tidak perlu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('Agg')

# Konfigurasi Path
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../dataset_preprocessed.csv")

def load_data():
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        print(f"Error membaca dataset di {dataset_path}: {e}")
        exit()

def create_dataset(dataset, look_back=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, :])
    return np.array(X), np.array(Y)

def train_and_export_model(df, group_name, columns, model_filename, scaler_filename, rep_col_idx):
    print(f"\n--- Memulai Pelatihan Model {group_name} ---")
    data = df[columns].values
    
    # 1. Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 2. Windowing (30 hari)
    look_back = 30
    X, y = create_dataset(scaled_data, look_back)
    
    # 3. Pembagian Data: Training (80%) dan Testing (20%)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 4. Desain Model LSTM (32 Units)
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(look_back, len(columns))))
    model.add(Dense(len(columns)))
    model.compile(optimizer='adam', loss='mse')
    
    # 5. Konfigurasi Early Stopping (Stop bila val_loss tidak membaik setelah 15 epoch)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    # 6. Training Model
    print(f"Melatih model {group_name} dengan {look_back} windows, 32 units, maks 100 epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 7. Evaluasi dan Rekam Hasil untuk visualisasi
    predictions = model.predict(X_test, verbose=0)
    preds_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # 8. Ekspor Models dan Scalers
    model_path = os.path.join(script_dir, model_filename)
    scaler_path = os.path.join(script_dir, scaler_filename)
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"[BERHASIL] Model {group_name} diekspor ke: {model_filename}")
    print(f"[BERHASIL] Scaler {group_name} diekspor ke: {scaler_filename}")
    
    # Return data for plotting later (representative column)
    col_name = columns[rep_col_idx].replace('_', ' ').title()
    if col_name == 'Daging Ayam':
        col_name = 'Daging Sapi (Mapping Ayam)'
        
    return {
        'group': group_name,
        'col_name': col_name,
        'actuals': y_test_inv[:, rep_col_idx],
        'predictions': preds_inv[:, rep_col_idx]
    }

if __name__ == '__main__':
    df = load_data()
    
    # Mapping kolom
    cols_fluktuatif = ['cabai_merah', 'cabai_rawit', 'bawang_merah', 'daging_ayam', 'telur_ayam']
    cols_stabil = ['beras', 'minyak_goreng', 'bawang_putih']
    
    # Train Fluktuatif (rep idx = 0: cabai_merah)
    res_fluktuatif = train_and_export_model(
        df, "Fluktuatif", cols_fluktuatif, 
        'model_fluktuatif.keras', 'scaler_fluktuatif.gz', 0
    )
    
    # Train Stabil (rep idx = 0: beras)
    res_stabil = train_and_export_model(
        df, "Stabil", cols_stabil, 
        'model_stabil.keras', 'scaler_stabil.gz', 0
    )
    
    # === MEMBUAT VISUALISASI FINAL ===
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    fig.suptitle('Evaluasi Final Model: Harga Asli vs Prediksi pada Data Testing', fontsize=16)
    
    # Plot Fluktuatif
    axes[0].plot(res_fluktuatif['actuals'], label='Harga Asli', color='blue')
    axes[0].plot(res_fluktuatif['predictions'], label='Harga Prediksi', color='red', linestyle='dashed')
    axes[0].set_title(f"Grup {res_fluktuatif['group']} (Perwakilan: {res_fluktuatif['col_name']})")
    axes[0].set_ylabel('Harga (Rupiah)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Stabil
    axes[1].plot(res_stabil['actuals'], label='Harga Asli', color='blue')
    axes[1].plot(res_stabil['predictions'], label='Harga Prediksi', color='red', linestyle='dashed')
    axes[1].set_title(f"Grup {res_stabil['group']} (Perwakilan: {res_stabil['col_name']})")
    axes[1].set_xlabel('Data Point (Testing 20%)')
    axes[1].set_ylabel('Harga (Rupiah)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(script_dir, "final_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Plot evaluasi final disimpan ke: {plot_path}")
    print("[SELESAI] Semua model telah dilatih dan diekspor. Server backend kini bisa dijalankan!")
