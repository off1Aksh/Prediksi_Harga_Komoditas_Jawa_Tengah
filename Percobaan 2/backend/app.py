import os
import numpy as np
import pandas as pd
import joblib
import json
import io
from PIL import Image
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

app = Flask(__name__)
# Enable CORS for all domains, important for React frontend to connect
CORS(app)

# Konfigurasi Path Model dan Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_univariate")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler_master.gz")

# Update dataset path ke dataset terbaru (9 tahun)
RAW_DATASET_PATH = os.path.join(BASE_DIR, "Dataset_2017_2026", "dataset_harga_komoditas_2017_2026.csv")

# Global variables
scaler_master = None
models = {} # Dictionary untuk menyimpan 8 model Univariate
vision_model = None
vision_class_names = []
df_raw = None        # Dataset dengan harga Rupiah asli (setelah LOCF)

# Urutan kolom data (sama dengan di dataset):
ORDERED_COLUMNS = [
    'bawang_merah', 'bawang_putih', 'beras', 'cabai_merah',
    'cabai_rawit', 'daging_ayam', 'minyak_goreng', 'telur_ayam'
]

def load_raw_dataset():
    """Membaca dataset asli, membersihkan, dan menerapkan LOCF."""
    global df_raw
    try:
        df = pd.read_csv(RAW_DATASET_PATH)
        # Konversi tanggal secara pintar
        parsed_dates = pd.to_datetime(df["tanggal"], format="%d/ %m/ %Y", errors="coerce")
        if parsed_dates.isnull().all():
            # Fallback ke format standar YYYY-MM-DD
            df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
        else:
            df["tanggal"] = parsed_dates
        
        # Hilangkan koma dan konversi ke float
        kolom_harga = [col for col in df.columns if col != "tanggal"]
        for col in kolom_harga:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)
        
        # LOCF - Forward Fill untuk tanggal yang hilang
        df = df.set_index("tanggal")
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_range)
        df.index.name = "tanggal"
        df = df.ffill()
        df_raw = df[ORDERED_COLUMNS] # Pastikan urutan selalu sesuai
        print(f"Dataset RAW berhasil dimuat: {df_raw.shape[0]} baris")
    except Exception as e:
        print(f"Error memuat dataset RAW: {e}")

def load_ai_assets():
    global scaler_master, models, vision_model, vision_class_names
    try:
        print(f"Loading Scaler Master dari: {SCALER_PATH}")
        scaler_master = joblib.load(SCALER_PATH)
        print("Scaler Master berhasil dimuat.")

        for col in ORDERED_COLUMNS:
            model_path = os.path.join(MODELS_DIR, f"model_{col}.keras")
            print(f"Loading Model Univariate: {col}")
            models[col] = load_model(model_path)
            
        print("Ke-8 Model Univariate berhasil dimuat secara terpisah.")
        
        # Load Vision Model
        VISION_MODEL_PATH = os.path.join(BASE_DIR, "models_vision", "vision_model.keras")
        CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models_vision", "class_names.json")
        if os.path.exists(VISION_MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
            vision_model = load_model(VISION_MODEL_PATH)
            with open(CLASS_NAMES_PATH, "r") as f:
                vision_class_names = json.load(f)
            print("Model Computer Vision berhasil dimuat.")

    except Exception as e:
        print(f"Error memuat model/scaler: {e}")

# Load assets saat aplikasi Flask dijalankan
load_raw_dataset()
load_ai_assets()

@app.route('/api/latest-data', methods=['GET'])
def get_latest_data():
    """
    Endpoint untuk mengambil 30 baris terakhir dari dataset dalam format RUPIAH ASLI.
    Juga mengembalikan tanggal untuk ditampilkan di frontend.
    """
    try:
        if df_raw is None:
            return jsonify({"error": "Dataset RAW belum berhasil dimuat."}), 500

        # Ambil 30 baris terakhir dari dataset asli (Rupiah)
        df_latest = df_raw.tail(30)

        # Format data: list of lists dengan harga Rupiah
        latest_data = df_latest[ORDERED_COLUMNS].values.tolist()

        # Ambil tanggal
        dates = [d.strftime('%Y-%m-%d') for d in df_latest.index]

        return jsonify({
            "historical_data": latest_data,
            "dates": dates
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi 7 hari ke depan menggunakan 8 Model Univariate.
    Mengembalikan: prediksi dalam format RUPIAH ASLI.
    """
    try:
        if not models or scaler_master is None:
            return jsonify({"error": "Model atau Scaler belum berhasil dimuat. Cek path file."}), 500

        if df_raw is None:
            return jsonify({"error": "Dataset RAW belum berhasil dimuat."}), 500

        # Ambil 30 baris terakhir dari dataset RUPIAH ASLI
        raw_data = df_raw[ORDERED_COLUMNS].tail(30).values.astype(float)

        if raw_data.shape != (30, 8):
            return jsonify({
                "error": f"Dimensi array tidak sesuai. Diharapkan (30, 8), tetapi mendapat {raw_data.shape}"
            }), 400

        # 1. Transform 30 baris ke skala 0-1 SEKALI saja
        scaled_data = scaler_master.transform(raw_data)
        
        # Tempat menyimpan prediksi akhir (7 hari x 8 komoditas)
        predictions_scaled = np.zeros((7, 8))

        # 2. Proses Iterative Forecasting (7 Hari) untuk MASING-MASING komoditas
        for idx, col in enumerate(ORDERED_COLUMNS):
            model = models[col]
            
            # Ambil data 30 hari historis (HANYA untuk komoditas ini) -> bentuknya (30,)
            current_window = scaled_data[:, idx].copy()
            
            # Prediksi maju 7 hari
            for day in range(7):
                # Reshape untuk input LSTM: (batch_size=1, window_size=30, features=1)
                in_data = current_window.reshape((1, 30, 1))
                
                # Prediksi nilai 1 hari ke depan (masih dalam skala 0-1)
                pred = model.predict(in_data, verbose=0)[0][0]
                
                # Simpan hasil prediksi
                predictions_scaled[day, idx] = pred
                
                # Geser window (Sliding Window): hapus hari tertua, tambahkan prediksi terbaru
                current_window = np.append(current_window[1:], pred)

        # 3. Inverse Transform: kembalikan seluruh hasil prediksi (skala 0-1) ke Rupiah asli
        final_forecast = scaler_master.inverse_transform(predictions_scaled)

        # 4. Bulatkan ke bilangan bulat
        final_forecast = np.round(final_forecast).astype(int)

        return jsonify({
            "message": "Prediksi 7 hari dengan 8 Model Univariate berhasil.",
            "forecast": final_forecast.tolist()
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-excel', methods=['POST'])
def upload_excel():
    global df_raw
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada file yang diunggah"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nama file kosong"}), 400

        # Baca excel mentah (tanpa header karena struktur PIHPS tidak beraturan)
        df_upload = pd.read_excel(file, header=None)
        
        # Cari baris pertama yang memiliki kata 'Komoditas' untuk mengunci baris tanggal
        header_row_idx = 0
        for idx, row in df_upload.iterrows():
            if 'Komoditas' in str(row.iloc[1]):
                header_row_idx = idx
                break

        # Ekstrak Tanggal mulai kolom ke-2 (index 2)
        raw_dates = df_upload.iloc[header_row_idx, 2:].values
        
        # Bersihkan spasi pada tanggal ("01/ 01/ 2026" -> "01/01/2026")
        cleaned_dates = [str(d).replace(" ", "") for d in raw_dates if pd.notna(d)]
        
        new_data_dict = {"tanggal": cleaned_dates}
        
        LABEL_TO_COL = {
            'Bawang Merah': 'bawang_merah',
            'Bawang Putih': 'bawang_putih',
            'Beras': 'beras',
            'Cabai Merah': 'cabai_merah',
            'Cabai Rawit': 'cabai_rawit',
            'Daging Ayam': 'daging_ayam',
            'Minyak Goreng': 'minyak_goreng',
            'Telur Ayam': 'telur_ayam'
        }
        
        found_commodities = set()
        
        # Cari baris ke-8 komoditas (persis label namanya)
        for idx, row in df_upload.iterrows():
            komoditas_name = str(row.iloc[1]).strip()
            
            if komoditas_name in LABEL_TO_COL:
                col_key = LABEL_TO_COL[komoditas_name]
                prices = row.iloc[2 : 2+len(cleaned_dates)].values
                
                cleaned_prices = []
                for p in prices:
                    if pd.isna(p) or str(p).strip() == "-" or str(p).strip() == "":
                        cleaned_prices.append(np.nan)
                    else:
                        cleaned_prices.append(float(str(p).replace(",", "")))
                
                new_data_dict[col_key] = cleaned_prices
                found_commodities.add(komoditas_name)
        
        if len(found_commodities) < 8:
            missing = set(LABEL_TO_COL.keys()) - found_commodities
            return jsonify({"error": f"Komoditas berikut tidak ditemukan atau nama tidak persis: {missing}"}), 400
            
        df_new = pd.DataFrame(new_data_dict)
        
        # Konversi tanggal. Karena format awal adalah DD/MM/YYYY
        df_new["tanggal"] = pd.to_datetime(df_new["tanggal"], format="%d/%m/%Y", errors="coerce")
        df_new = df_new.dropna(subset=["tanggal"])
        df_new = df_new.set_index("tanggal")
        
        # Gabungkan dan rapikan ke df_raw global
        df_combined = pd.concat([df_raw, df_new])
        df_combined = df_combined.sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        
        full_range = pd.date_range(start=df_combined.index.min(), end=df_combined.index.max(), freq="D")
        df_combined = df_combined.reindex(full_range)
        df_combined.index.name = "tanggal"
        df_combined = df_combined.ffill()
        df_combined = df_combined[ORDERED_COLUMNS]
        
        # Timpa data di memori
        df_raw = df_combined
        
        # Save permanen ke CSV agar aman saat restart
        df_to_save = df_raw.copy()
        # Ekspor index dengan format standar internasional YYYY-MM-DD
        df_to_save.index = df_to_save.index.strftime('%Y-%m-%d')
        df_to_save.to_csv(RAW_DATASET_PATH)
        
        return jsonify({"message": "Data Excel berhasil diekstrak dan digabungkan ke database!", "added_rows": len(df_new)}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses Excel: {str(e)}"}), 500

@app.route('/api/predict-image', methods=['POST'])
def predict_image():
    """
    Endpoint untuk mendeteksi gambar komoditas dan langsung meramalkan harganya.
    """
    try:
        if vision_model is None:
            return jsonify({"error": "Vision Model belum dimuat oleh server."}), 500

        if 'image' not in request.files:
            return jsonify({"error": "Tidak ada file gambar yang diunggah"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Nama file kosong"}), 400

        # Membaca gambar ke dalam memori
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Prediksi Gambar (Tidak perlu manual preprocess_input karena sudah tertanam di dalam model)
        preds = vision_model.predict(x)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        detected_class = vision_class_names[class_idx]
        
        # Jika tingkat keyakinan terlalu rendah (kurang dari 40%)
        if confidence < 0.4:
            return jsonify({
                "error": "Gambar tidak dapat dikenali dengan pasti. Silakan foto ulang komoditas dengan lebih jelas.",
                "confidence": confidence
            }), 400

        # Jika berhasil ditebak, panggil LSTM untuk meramalkan harganya
        # Ambil data 30 hari terakhir dari df_raw
        raw_data = df_raw[ORDERED_COLUMNS].tail(30).values.astype(float)
        scaled_data = scaler_master.transform(raw_data)
        
        idx = ORDERED_COLUMNS.index(detected_class)
        model_lstm = models[detected_class]
        
        current_window = scaled_data[:, idx].copy()
        predictions_scaled = np.zeros((7, 8))
        
        for day in range(7):
            in_data = current_window.reshape((1, 30, 1))
            pred = model_lstm.predict(in_data, verbose=0)[0][0]
            predictions_scaled[day, idx] = pred
            current_window = np.append(current_window[1:], pred)
            
        final_forecast = scaler_master.inverse_transform(predictions_scaled)
        final_forecast = np.round(final_forecast).astype(int)
        
        # Ekstrak harga 7 hari HANYA untuk komoditas yang difoto
        commodity_forecast = final_forecast[:, idx].tolist()
        
        # Rapikan nama komoditas (misal: 'cabai_rawit' -> 'Cabai Rawit')
        display_name = detected_class.replace("_", " ").title()
        
        # Ambil informasi tanggal dan harga aktual terakhir
        import datetime
        last_date_obj = df_raw.index[-1]
        last_price = float(df_raw.iloc[-1][detected_class])
        
        last_date_str = last_date_obj.strftime('%d %b %Y')
        next_dates = [(last_date_obj + datetime.timedelta(days=i)).strftime('%d %b %Y') for i in range(1, 8)]
        
        return jsonify({
            "detected_commodity": detected_class,
            "display_name": display_name,
            "confidence": confidence,
            "forecast": commodity_forecast,
            "last_price": int(last_price) if not pd.isna(last_price) else 0,
            "last_date": last_date_str,
            "next_dates": next_dates
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500

@app.route('/api/predict-single/<komoditas>', methods=['GET'])
def predict_single(komoditas):
    """
    Endpoint untuk meramalkan harga berdasarkan pilihan komoditas manual.
    """
    try:
        if komoditas not in ORDERED_COLUMNS:
            return jsonify({"error": "Komoditas tidak ditemukan"}), 400

        # Ambil data 30 hari terakhir dari df_raw
        raw_data = df_raw[ORDERED_COLUMNS].tail(30).values.astype(float)
        scaled_data = scaler_master.transform(raw_data)
        
        idx = ORDERED_COLUMNS.index(komoditas)
        model_lstm = models[komoditas]
        
        current_window = scaled_data[:, idx].copy()
        predictions_scaled = np.zeros((7, 8))
        
        for day in range(7):
            in_data = current_window.reshape((1, 30, 1))
            pred = model_lstm.predict(in_data, verbose=0)[0][0]
            predictions_scaled[day, idx] = pred
            current_window = np.append(current_window[1:], pred)
            
        final_forecast = scaler_master.inverse_transform(predictions_scaled)
        final_forecast = np.round(final_forecast).astype(int)
        
        commodity_forecast = final_forecast[:, idx].tolist()
        display_name = komoditas.replace("_", " ").title()
        
        import datetime
        last_date_obj = df_raw.index[-1]
        last_price = float(df_raw.iloc[-1][komoditas])
        
        last_date_str = last_date_obj.strftime('%d %b %Y')
        next_dates = [(last_date_obj + datetime.timedelta(days=i)).strftime('%d %b %Y') for i in range(1, 8)]
        
        return jsonify({
            "detected_commodity": komoditas,
            "display_name": display_name,
            "confidence": 1.0, # Karena dipilih manual
            "forecast": commodity_forecast,
            "last_price": int(last_price) if not pd.isna(last_price) else 0,
            "last_date": last_date_str,
            "next_dates": next_dates
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses prediksi manual: {str(e)}"}), 500

if __name__ == '__main__':
    # Pastikan file dijalankan di port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
