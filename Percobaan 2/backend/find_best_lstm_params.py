import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import time

# Konfigurasi Direktori
DATA_PATH = r"d:\KULIAH\SKRIPSI\Percobaan 2\backend\Dataset_2017_2026\dataset_harga_komoditas_2017_2026.csv"
OUTPUT_DIR = r"d:\KULIAH\SKRIPSI\Percobaan 2\backend\Dataset_2017_2026\hasil_tuning"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "grafik")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Parameter Grid Search
SPLIT_RATIOS = [0.7, 0.8, 0.9]
WINDOW_SIZES = [7, 14, 30]
LSTM_UNITS = [32, 64, 128]

EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 10

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def run_grid_search():
    df = pd.read_csv(DATA_PATH, parse_dates=['tanggal'], index_col='tanggal')
    komoditas_list = df.columns.tolist()
    
    all_results = []
    
    total_combinations = len(komoditas_list) * len(SPLIT_RATIOS) * len(WINDOW_SIZES) * len(LSTM_UNITS)
    current_iteration = 0
    
    start_time_total = time.time()

    for komoditas in komoditas_list:
        print(f"\n{'='*50}\nMemulai Tuning untuk: {komoditas}\n{'='*50}")
        
        # Buat folder grafik khusus komoditas ini
        komoditas_graph_dir = os.path.join(GRAPH_DIR, komoditas)
        os.makedirs(komoditas_graph_dir, exist_ok=True)
        
        data_komoditas = df[[komoditas]].values.astype('float32')
        
        for split in SPLIT_RATIOS:
            train_size = int(len(data_komoditas) * split)
            train_data_raw, test_data_raw = data_komoditas[0:train_size], data_komoditas[train_size:len(data_komoditas)]
            
            # Skala masing-masing split untuk mencegah data leakage
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler.fit_transform(train_data_raw)
            test_scaled = scaler.transform(test_data_raw)
            
            for window in WINDOW_SIZES:
                X_train, y_train = create_dataset(train_scaled, window)
                X_test, y_test = create_dataset(test_scaled, window)
                
                # Reshape for LSTM [samples, time steps, features]
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                for units in LSTM_UNITS:
                    current_iteration += 1
                    print(f"[{current_iteration}/{total_combinations}] {komoditas} | Split: {split} | Window: {window} | Units: {units}")
                    
                    # Build Model
                    model = Sequential()
                    model.add(LSTM(units, return_sequences=False, input_shape=(window, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
                    
                    # Train
                    history = model.fit(
                        X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        verbose=0,
                        callbacks=[early_stop]
                    )
                    
                    stopped_epoch = early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else EPOCHS
                    
                    # Predict
                    train_predict = model.predict(X_train, verbose=0)
                    test_predict = model.predict(X_test, verbose=0)
                    
                    # Invert predictions
                    train_predict_inv = scaler.inverse_transform(train_predict)
                    y_train_inv = scaler.inverse_transform([y_train])
                    test_predict_inv = scaler.inverse_transform(test_predict)
                    y_test_inv = scaler.inverse_transform([y_test])
                    
                    # Metrics calculation
                    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict_inv[:,0]))
                    mae = mean_absolute_error(y_test_inv[0], test_predict_inv[:,0])
                    mape = mean_absolute_percentage_error(y_test_inv[0], test_predict_inv[:,0]) * 100
                    
                    # Save metrics
                    all_results.append({
                        'Komoditas': komoditas,
                        'Train_Split': split,
                        'Window_Size': window,
                        'LSTM_Units': units,
                        'RMSE': round(rmse, 2),
                        'MAE': round(mae, 2),
                        'MAPE (%)': round(mape, 2),
                        'Epoch_Berhenti': stopped_epoch
                    })
                    
                    # Save Plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(y_test_inv[0], label='Aktual', color='blue')
                    plt.plot(test_predict_inv[:,0], label='Prediksi', color='red')
                    plt.title(f"{komoditas} | Split: {split} | W: {window} | Units: {units}\nMAPE: {mape:.2f}%")
                    plt.xlabel('Hari (Data Testing)')
                    plt.ylabel('Harga (Rp)')
                    plt.legend()
                    plt.grid(True)
                    
                    plot_filename = f"S{int(split*100)}_W{window}_L{units}.png"
                    plot_path = os.path.join(komoditas_graph_dir, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

    # Ekspor Hasil Akhir
    results_df = pd.DataFrame(all_results)
    
    # Sortir dari yang terbaik (MAPE terendah) ke terburuk
    results_df = results_df.sort_values(by=['Komoditas', 'MAPE (%)'], ascending=[True, True])
    
    csv_path = os.path.join(OUTPUT_DIR, "semua_hasil_tuning.csv")
    results_df.to_csv(csv_path, index=False)
    
    end_time_total = time.time()
    elapsed = end_time_total - start_time_total
    
    print(f"\n{'='*50}")
    print(f"SELESAI! Waktu eksekusi total: {elapsed/60:.2f} menit")
    print(f"Hasil ringkasan disimpan di: {csv_path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    run_grid_search()
