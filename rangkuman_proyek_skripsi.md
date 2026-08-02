# 📋 Rangkuman Lengkap Proyek Skripsi (Terbaru)
## Implementasi Model LSTM untuk Prediksi Harga Komoditas Pangan Berbasis Web di Provinsi Jawa Tengah

**Nama Mahasiswa:** Akasha Bin Ali  
**NIM:** 4.33.22.0.01  
**Program Studi:** S.Tr Teknologi Rekayasa Komputer  
**Politeknik Negeri Semarang**  
**Lokasi Proyek:** `d:\KULIAH\SKRIPSI\Percobaan 2\`  
**Tanggal Update Terakhir:** 24 Juli 2026

---

## 1. Deskripsi Umum Proyek

Proyek skripsi ini membangun **Sistem Cerdas PanganPredict**, sebuah ekosistem aplikasi berbasis *Artificial Intelligence* (AI) yang mampu **mendeteksi jenis komoditas pangan dari gambar** dan **meramalkan harga komoditas tersebut untuk 7 hari ke depan**.

Sistem ini didukung oleh **dua model AI utama**:
1. **Model Computer Vision (MobileNetV2):** Mengenali gambar komoditas pangan dari kamera/galeri pengguna.
2. **Model Regresi Deret Waktu (Univariate LSTM):** Terdiri dari 8 model terpisah (masing-masing komoditas memiliki konfigurasi hyperparameter optimalnya sendiri) untuk meramalkan harga berdasarkan data historis 9 tahun.

Proyek ini terintegrasi secara **full-stack** yang terdiri dari:
- **Backend:** Flask Server (Python) — API REST untuk prediksi, deteksi gambar, riwayat harga, dan upload data baru.
- **Frontend Web:** React 19 + Vite 7 + Tailwind CSS 4 + React Router v7 — Dashboard analis profesional.
- **Frontend Mobile:** React Native 0.81 + Expo 54 — Aplikasi mobile dengan kamera dan prediksi langsung.

### 8 Komoditas yang Didukung:
Bawang Merah, Bawang Putih, Beras, Cabai Merah, Cabai Rawit, Daging Ayam, Minyak Goreng, Telur Ayam.

---

## 2. Sumber dan Struktur Data

### 2.1. Dataset Harga (Time Series)
- **Sumber:** Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional.
- **Wilayah:** Provinsi Jawa Tengah.
- **Periode:** 2017 hingga 2026 (9 tahun).
- **Format Asli:** File Excel dari PIHPS (format tanggal `DD/ MM/ YYYY`, harga menggunakan koma sebagai pemisah ribuan).
- **File Dataset Utama (setelah preprocessing):** `Dataset_2017_2026/dataset_harga_komoditas_2017_2026.csv`
  - Kolom: `tanggal`, `bawang_merah`, `bawang_putih`, `beras`, `cabai_merah`, `cabai_rawit`, `daging_ayam`, `minyak_goreng`, `telur_ayam`.
  - Total data (setelah LOCF): ~3.400+ baris (harian kontinu).
- **File Dataset Preprocessed (ternormalisasi):** `Dataset_2017_2026/dataset_preprocessed.csv`

### 2.2. Dataset Gambar (Computer Vision)
- **Lokasi:** `d:\KULIAH\SKRIPSI\Dataset\gambar_balanced\`
- **Struktur:** 8 folder (satu per komoditas), masing-masing berisi tepat 100 gambar.
- **Total Data:** 800 gambar (Balanced Dataset).
- **Sumber Gambar:** Dikumpulkan dari internet dan foto manual (beras menggunakan foto kondisi tumpukan/realistis).
- **Script Preparasi:** `balance_dataset.py` — melakukan random undersampling agar jumlah data tiap kelas seimbang.

### 2.3. Dataset Harga Asli (sebelum penggabungan)
Tersimpan di `d:\KULIAH\SKRIPSI\Dataset\harga\` sebagai file Excel terpisah per komoditas dan per periode:
- `bawang_merah_putih_2017_2020.xlsx`, `bawang_merah_putih_2021_2026.xlsx`
- `beras_2017_2020.xlsx`, `beras 2021-2026.xlsx`
- `cabai merah dan cabe rawit 2017-2020.xlsx`, `cabai merah dan cabe rawit 2021-2026.xlsx`
- `daging ayam 2017-2020.xlsx`, `daging ayam 2021-2026.xlsx`
- `minyak goreng 2017-2020.xlsx`, `minyak goreng 2021-2026.xlsx`
- `telur ayam 2017-2020.xlsx`, `telur ayam 2021-2026.xlsx`

---

## 3. Tahapan Pra-pemrosesan Data (Data Pre-processing)

Dilakukan melalui script `pre-processing.py`:

### 3.1. Konversi Tipe Data
- Kolom tanggal dikonversi ke `datetime` dengan format `%d/ %m/ %Y`.
- Kolom harga dibersihkan dari koma pemisah ribuan dan dikonversi ke `float`.

### 3.2. LOCF (Last Observation Carried Forward)
- Tanggal di-set sebagai index lalu dibuat `date_range` harian lengkap.
- Tanggal yang hilang (weekend/libur PIHPS) diisi menggunakan `ffill()` (forward fill) — metode LOCF.
- Contoh: Jika data Sabtu dan Minggu tidak ada, harga Jumat akan dipakai.

### 3.3. Normalisasi Min-Max
- Menggunakan `MinMaxScaler(feature_range=(0, 1))` dari scikit-learn.
- Scaler di-fit pada **seluruh 8 kolom harga secara bersamaan** (scaler global/master) sehingga semua komoditas dinormalisasi dengan skala yang konsisten.
- Scaler disimpan sebagai `scaler_master.gz` via joblib untuk dipakai saat inferensi.

### 3.4. Metode Sliding Window
- Data deret waktu diubah menjadi pasangan (X, Y) untuk supervised learning.
- X = sekuens harga sepanjang `window_size` hari, Y = harga 1 hari berikutnya.
- Variasi window yang diuji: **7, 14, dan 30 hari**.

### 3.5. Pembagian Data (Sequential Holdout / Time-Series Split)
- Data dibagi secara **kronologis** (bukan acak) untuk menjaga urutan temporal.
- Rasio split yang diuji pada fase tuning: **70/30, 80/20, 90/10**.
- Rasio split pada eksperimen skenario utama: **80% Training / 20% Testing**.

---

## 4. Arsitektur AI (Dua Model Utama)

### 4.1. Model Prediksi Harga — Univariate LSTM

**Pendekatan:** Univariate — setiap komoditas dimodelkan secara independen dengan arsitektur dan hyperparameter optimalnya masing-masing.

**Arsitektur Dasar:**
```
Input (window_size, 1) → LSTM(units, activation='tanh') → Dense(1)
```

- **Layer LSTM:** Single layer (tidak di-stack), `return_sequences=False`.
- **Fungsi Aktivasi LSTM:** `tanh` (default standar LSTM).
- **Layer Output:** `Dense(1)` dengan aktivasi linear (regresi).
- **Optimizer:** Adam.
- **Loss Function:** Mean Squared Error (MSE).
- **Batch Size:** 32.

**Konfigurasi Hyperparameter Optimal Per Komoditas** (hasil dari Sequential Holdout Tuning dengan 216 kombinasi):

| Komoditas | Window Size | LSTM Units |
|---|:---:|:---:|
| Bawang Merah | 30 | 64 |
| Bawang Putih | 7 | 128 |
| Beras | 14 | 128 |
| Cabai Merah | 30 | 128 |
| Cabai Rawit | 30 | 32 |
| Daging Ayam | 30 | 64 |
| Minyak Goreng | 30 | 128 |
| Telur Ayam | 30 | 64 |

> **Catatan Penting:** Setiap komoditas memiliki konfigurasi hyperparameter yang berbeda-beda, hasil dari proses tuning individual menggunakan Sequential Holdout. Ini berbeda dari eksperimen skenario (yang menggunakan konfigurasi seragam untuk semua komoditas).

### 4.2. Model Deteksi Gambar — MobileNetV2 (Transfer Learning)

Model ini dikembangkan melalui proses **Hyperparameter Tuning** menggunakan **Stratified 5-Fold Cross-Validation (Control Variable Method)** dengan 6 tahap pengujian (Baseline, LR, Batch Size, Dropout, Epochs, Augmentation).

**Arsitektur (Konfigurasi Terbaik):**
```
Input (224, 224, 3) 
→ Data Augmentation (RandomFlip Horizontal)
→ MobileNetV2 Preprocessing (rescale ke [-1, 1])
→ MobileNetV2 Base (ImageNet weights, FROZEN / tidak ikut dilatih)
→ GlobalAveragePooling2D()
→ Dropout(0.3)
→ Dense(8, activation='softmax')
```

**Parameter Training Final:**

| Parameter | Nilai |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | SparseCategoricalCrossentropy |
| Metrik | Accuracy |
| Epochs | 50 (max) |
| Batch Size | 64 |
| Train/Val Split | 80/20 (seed=123) |
| Early Stopping | monitor='val_loss', patience=5 |
| Dataset Optimization | `.cache().shuffle(1000).prefetch(AUTOTUNE)` |
| Confidence Threshold | < 0.4 → ditolak ("Gambar tidak dapat dikenali") |
| Hasil Evaluasi | Accuracy: 95.48%, F1-Score: 95.60% |

**File Output (Final):**
- `models_vision/vision_model.keras` (~9.7 MB)
- `models_vision/class_names.json` (mapping 8 kelas)
- `models_vision/confusion_matrix.png` (evaluasi tebakan per kelas)
- `models_vision/classification_report.txt` (metrik detail Presisi/Recall)
- `models_vision/training_history.png` (kurva training stabil, early stop di epoch 41)

---

## 5. Eksperimen dan Evaluasi

### 5.1. Eksperimen Skenario (6 Skenario — Konfigurasi Seragam)

Script: `eksperimen_skripsi.py`

Pengujian mendalam dilakukan dengan **6 skenario kombinasi** Window Size dan LSTM Units. Semua komoditas menggunakan konfigurasi **yang sama** dalam setiap skenario (untuk perbandingan adil).

**Parameter Training Eksperimen:**
- Epochs: **1000** (max)
- Batch Size: **32**
- Early Stopping: `monitor='val_loss'`, `patience=50`, `start_from_epoch=50`, `restore_best_weights=True`
- Activation LSTM: `tanh`
- Split Data: 80/20 (kronologis)
- Reproducibility: `np.random.seed(42)`, `tf.random.set_seed(42)`

**Hasil Rekapitulasi:**

| Skenario | Window Size | LSTM Units | Rata-rata RMSE (Rp) | Rata-rata MAPE (%) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 7 | 32 | 636,24 | 1,02% |
| 2 | 7 | 64 | 637,97 | 1,05% |
| 3 | 14 | 32 | 618,50 | 1,01% |
| 4 | 14 | 64 | 602,56 | 1,00% |
| 5 | 30 | 32 | 593,04 | 0,88% |
| **6** 🏆 | **30** | **64** | **591,90** | **0,86%** |

**Hasil Detail Per Komoditas (Skenario 6 — Terbaik):**

| Komoditas | RMSE (Rp) | MAE (Rp) | MAPE (%) | Epoch Berhenti |
|---|:---:|:---:|:---:|:---:|
| Bawang Merah | 793,82 | 398,49 | 0,97% | 84 |
| Bawang Putih | 359,12 | 150,32 | 0,36% | 51 |
| Beras | 54,39 | 43,72 | 0,30% | 51 |
| Cabai Merah | 1.369,21 | 815,03 | 1,93% | 51 |
| Cabai Rawit | 1.719,77 | 1.003,81 | 2,43% | 51 |
| Daging Ayam | 229,19 | 155,49 | 0,44% | 51 |
| Minyak Goreng | 36,69 | 20,76 | 0,11% | 118 |
| Telur Ayam | 173,05 | 103,36 | 0,38% | 51 |

**Output per Skenario:**
- `evaluasi_Skenario_X_WY_LZ.csv` — hasil evaluasi per komoditas.
- `grafik_prediksi_vs_aktual.png` — plot 100 hari terakhir data testing (8 subplot, 1 per komoditas).
- `grafik_loss_history.png` — kurva training loss vs validation loss (8 subplot).
- Folder `models/` — 8 file model `.keras` hasil training skenario tersebut.

### 5.2. Sequential Holdout Tuning (216 Kombinasi — Hyperparameter Per Komoditas)

Script: `find_best_lstm_params.py`

Ini merupakan **eksperimen inti yang menentukan konfigurasi model produksi**. Pencarian parameter terbaik dilakukan untuk **masing-masing komoditas secara individual** menggunakan grid search:
- **Split Ratios:** [0.7, 0.8, 0.9] (70/30, 80/20, 90/10)
- **Window Sizes:** [7, 14, 30] hari
- **LSTM Units:** [32, 64, 128] unit
- **Total:** 3 × 3 × 3 = **27 kombinasi per komoditas** × 8 komoditas = **216 run total**.

**Parameter Training:**
- Epochs: **100** (max)
- Batch Size: **16**
- Early Stopping: `monitor='val_loss'`, `patience=10`, `restore_best_weights=True`
- Optimizer: Adam
- Loss: MSE
- Scaler: **Per-komoditas** (fit pada train, transform pada test — **mencegah data leakage**)

**Hasil Terbaik Per Komoditas (Dipilih berdasarkan MAPE terendah):**

| Komoditas | Split | Window Size | LSTM Units | RMSE (Rp) | MAE (Rp) | MAPE (%) | Epoch Berhenti |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bawang Merah | 0.9 | 30 | 64 | 608,77 | 383,27 | **0,85%** | 57 |
| Bawang Putih | 0.9 | 7 | 128 | 181,81 | 110,44 | **0,28%** | 41 |
| Beras | 0.9 | 14 | 128 | 24,38 | 11,86 | **0,08%** | 29 |
| Cabai Merah | 0.9 | 30 | 128 | 1.214,40 | 699,28 | **1,57%** | 46 |
| Cabai Rawit | 0.9 | 30 | 32 | 1.577,77 | 888,08 | **2,10%** | 92 |
| Daging Ayam | 0.8 | 30 | 64 | 209,61 | 126,80 | **0,36%** | 44 |
| Minyak Goreng | 0.9 | 30 | 128 | 29,31 | 16,41 | **0,08%** | 40 |
| Telur Ayam | 0.9 | 30 | 64 | 132,88 | 79,74 | **0,29%** | 37 |

> **Catatan:** Konfigurasi terbaik dari tabel di atas inilah yang diadopsi sebagai `BEST_PARAMS` pada model produksi (`train_univariate_final.py` dan `app.py`).

**Output:**
- `hasil_tuning/semua_hasil_tuning.csv` — seluruh 216 hasil evaluasi lengkap.
- `hasil_tuning/grafik/<komoditas>/S{split}_W{window}_L{units}.png` — 216 plot perbandingan aktual vs prediksi (27 grafik per komoditas).

### 5.3. Percobaan Multivariate (Grup Fluktuatif & Stabil)

Script: `train_final_models.py`

Percobaan alternatif yang mengelompokkan komoditas berdasarkan karakteristik volatilitas:
- **Grup Fluktuatif (5 komoditas):** Cabai Merah, Cabai Rawit, Bawang Merah, Daging Ayam, Telur Ayam.
- **Grup Stabil (3 komoditas):** Beras, Minyak Goreng, Bawang Putih.

**Arsitektur:** LSTM(32, activation='relu', input_shape=(30, num_columns)) → Dense(num_columns)  
**Parameter:** Epochs=100, Batch=16, EarlyStopping patience=15  
**Kesimpulan:** Pendekatan multivariate ini menghasilkan *noise interference* antar komoditas. **Pendekatan Univariate (8 model terpisah) terbukti lebih unggul** dan digunakan sebagai arsitektur final.

---

## 6. Training Model Produksi (Final)

Script: `train_univariate_final.py`

Setelah menentukan hyperparameter terbaik per komoditas (dari Sequential Holdout Tuning), model produksi dilatih menggunakan **100% seluruh data** (tanpa validation split) untuk menyerap memori harga hingga hari terakhir.

**Konfigurasi:**
- **Epochs:** 1500 (max)
- **Batch Size:** 32
- **Early Stopping:** `monitor='loss'`, `patience=20`, `start_from_epoch=100` (karena tidak ada data validasi)
- **Scaler:** MinMaxScaler global di-fit pada seluruh 8 kolom → disimpan sebagai `scaler_master.gz`
- **Reproducibility:** `np.random.seed(42)`, `tf.random.set_seed(42)`
- **Hyperparameter:** Kustom per komoditas (sesuai tabel `BEST_PARAMS`)

**File Output Produksi:**
```
models_univariate/
├── model_bawang_merah.keras   (227 KB — LSTM 64 units)
├── model_bawang_putih.keras   (824 KB — LSTM 128 units)
├── model_beras.keras          (824 KB — LSTM 128 units)
├── model_cabai_merah.keras    (824 KB — LSTM 128 units)
├── model_cabai_rawit.keras    (76 KB  — LSTM 32 units)
├── model_daging_ayam.keras    (227 KB — LSTM 64 units)
├── model_minyak_goreng.keras  (824 KB — LSTM 128 units)
├── model_telur_ayam.keras     (227 KB — LSTM 64 units)
└── scaler_master.gz           (681 B  — Global MinMaxScaler)
```

> Ukuran file model berbeda-beda karena jumlah LSTM Units yang berbeda per komoditas (32 vs 64 vs 128 units).

---

## 7. Implementasi Sistem Produksi (Full-Stack)

### 7.1. Backend Flask — `app.py` (520+ baris)

**Dependensi Utama:** Flask, flask-cors, tensorflow, numpy, pandas, scikit-learn, joblib, Pillow, PyJWT, python-dotenv.

**Proses Startup:**
1. Memuat dataset CSV → parsing tanggal → LOCF → simpan ke `df_raw` global.
2. Memuat 8 model LSTM dari `models_univariate/`.
3. Memuat `scaler_master.gz`.
4. Memuat `vision_model.keras` dan `class_names.json`.

**Mekanisme Prediksi — Iterative Autoregressive Forecasting (7 Hari):**
1. Ambil 30 baris terakhir dari `df_raw` (8 kolom harga Rupiah asli).
2. Transform ke skala 0-1 menggunakan `scaler_master`.
3. Untuk **setiap komoditas secara terpisah**:
   - Ambil `window_size` hari terakhir (sesuai `BEST_PARAMS` komoditas tersebut).
   - Loop 7 kali (H+1 sampai H+7):
     - Reshape input ke `(1, window_size, 1)`.
     - Prediksi 1 langkah ke depan.
     - Geser window: hapus hari tertua, tambahkan prediksi terbaru.
4. Inverse transform seluruh matriks prediksi `(7, 8)` kembali ke Rupiah.
5. Bulatkan ke bilangan bulat.

**API Endpoints:**

| Endpoint | Method | Fungsi |
|---|:---:|---|
| `/api/login` | POST | Autentikasi Admin (menghasilkan JWT Token berlaku 24 jam) |
| `/api/latest-data` | GET | Mengambil 30 baris terakhir data harga Rupiah asli + tanggal |
| `/api/predict` | POST | Prediksi 7 hari ke depan untuk **semua 8 komoditas** sekaligus |
| `/api/predict-image` | POST | Terima gambar → deteksi MobileNetV2 → prediksi LSTM 7 hari (komoditas terdeteksi) |
| `/api/predict-single/<komoditas>` | GET | Prediksi 7 hari untuk komoditas yang dipilih manual (tanpa gambar) |
| `/api/history` | GET | Ambil data historis per komoditas, bulan, dan tahun |
| `/api/upload-excel` | POST | **[Protected by JWT]** Upload file Excel PIHPS → parsing → gabungkan ke dataset |

**Detail `/api/predict-image` Response:**
```json
{
  "detected_commodity": "beras",
  "display_name": "Beras",
  "confidence": 0.9523,
  "forecast": [15000, 15100, 15050, 15200, 15150, 15300, 15250],
  "last_price": 14800,
  "last_date": "24 Jul 2026",
  "next_dates": ["25 Jul 2026", "26 Jul 2026", "..."],
  "historical_prices": ["... 30 harga terakhir ..."],
  "historical_dates": ["... 30 tanggal terakhir ..."]
}
```

**Detail `/api/upload-excel` (Fitur Update Data Dinamis):**
- Menerima file Excel PIHPS mentah.
- Mencari baris header "Komoditas" secara otomatis.
- Mengekstrak tanggal dan harga 8 komoditas.
- Menggabungkan dengan dataset yang ada (duplikat ditimpa dengan data terbaru).
- Menerapkan LOCF ulang.
- Menyimpan secara permanen ke CSV.

### 7.2. Frontend Web — React 19 + Vite 7 + Tailwind CSS 4

**Dependensi Utama:** `react` (19.2), `react-dom` (19.2), `react-router-dom` (7.14), `recharts` (3.8), `lucide-react` (0.577), `axios` (1.13), `tailwindcss` (4.2), `framer-motion` (12.0).

**Arsitektur:** Multi-page SPA dengan React Router v7. Dilengkapi dengan animasi *page transition* menggunakan Framer Motion.

**Halaman & Routing:**

| Route | Komponen | Fungsi |
|---|---|---|
| `/` | `Home.jsx` | Landing page hero section |
| `/dashboard` | `Dashboard.jsx` | Dashboard utama: prediksi, grafik, upload |
| `/about` | `About.jsx` | Informasi teknis tentang model AI |

**Komponen Shared:**
- `Navbar.jsx` — Navigasi atas dengan highlight rute aktif.
- `Footer.jsx` — Footer dengan copyright dan label arsitektur.

**Dashboard.jsx — Fitur Utama (UI Premium & Animasi):**
1. **Login Admin Modal:** *Popup* otentikasi beranimasi menggunakan `AnimatePresence`. Mengamankan fitur upload excel.
2. **Upload Excel PIHPS (Protected):** Tombol upload data mentah dengan validasi JWT Token.
3. **8 Kartu Komoditas Premium:** 
   - Desain transparan elegan (Glassmorphism ringan).
   - Animasi *bouncy tap & hover* menggunakan `framer-motion`.
   - Gambar asli komoditas (overlapping asset) di sudut kanan.
4. **Grafik Dual-Area (Recharts AreaChart):**
   - 30 hari historis (Area hijau emerald dengan gradient) + 7 hari prediksi (Area amber dengan gradient).
   - Tooltip hover dengan format Rupiah Indonesia.
   - Dropdown pemilih komoditas di dalam header grafik.
5. **Tabel Prediksi & Badge 7 Hari:**
   - Desain *badge* interaktif (↗ hijau muda jika harga turun / ideal, ↘ merah jika harga naik).
   - Format Rupiah dengan `Intl.NumberFormat('id-ID')`.

**Backend Endpoints yang Digunakan:**
- `GET /api/latest-data` — saat halaman dimuat.
- `POST /api/predict` — setelah data dimuat / setelah upload.
- `POST /api/upload-excel` — saat user mengupload file Excel.

### 7.3. Mobile App — React Native 0.81 + Expo 54

**Dependensi Utama:** `react` (19.1), `react-native` (0.81.5), `expo` (54), `expo-camera` (17), `expo-image-picker` (17), `axios` (1.17), `lucide-react-native` (1.17), `react-native-chart-kit` (7), `react-native-svg` (15.12).

**Arsitektur:** Single-file App (`App.js`, ~624 baris) dengan navigasi state-based (tanpa library navigasi eksternal).

**Navigasi:**
- **Sidebar Drawer:** Custom slide-in modal dengan animasi, diaktifkan via tombol hamburger.
- **Layar Utama:** 2 screen — `'home'` (Scan/Prediksi) dan `'history'` (Riwayat Harga).

**Fitur Layar Home (Scan & Prediksi):**
1. **Kamera:** Menggunakan `expo-camera` — foto komoditas langsung.
2. **Galeri:** Menggunakan `expo-image-picker` — pilih gambar dari HP.
3. **Pemilihan Manual:** Modal pilihan komoditas tanpa perlu mengunggah gambar.
4. **Hasil Prediksi:**
   - Nama komoditas + confidence score.
   - Harga terakhir dengan tanggal aktual.
   - Tabel 7 hari harga prediksi dengan tanggal kalender.
   - Grafik gabungan: 30 hari historis (hijau) + 7 hari prediksi (biru) — `LineChart` dengan bezier curve.

**Fitur Layar Riwayat Harga:**
1. **Filter:** Dropdown komoditas, bulan (Jan-Des), dan tahun (2017-2026).
2. **Grafik Riwayat:** LineChart horizontal-scroll dengan Y-axis statis (overlay trick).
3. **Statistik Ringkasan:** Harga terendah, tertinggi, rata-rata.

**Backend Endpoints yang Digunakan:**
- `POST /api/predict-image` — setelah memotret/memilih gambar.
- `GET /api/predict-single/<komoditas>` — setelah memilih komoditas secara manual.
- `GET /api/history` — saat menelusuri riwayat harga.

**API Base URL:** `http://192.168.1.63:5000` (koneksi jaringan lokal).

---

## 8. Diagram dan Dokumentasi Visual

Tersimpan di `d:\KULIAH\SKRIPSI\Laporan\Diagrams\`:

| Diagram | Format |
|---|---|
| Arsitektur LSTM | SVG |
| Desain Arsitektur Model CNN (MobileNetV2) | SVG + DrawIO |
| Diagram Alir Metode Pengembangan Waterfall | SVG |
| Diagram Alir Sistem | SVG + DrawIO |
| Diagram Use Case | SVG + DrawIO |
| Flowchart Preprocessing Data | SVG + DrawIO |
| Flowchart Tahapan LSTM | SVG + DrawIO |
| Ilustrasi Sliding Window 30 | SVG + DrawIO |
| Perancangan Arsitektur Sistem | SVG + DrawIO |
| UML Activity Diagram | SVG |

---

## 9. Metrik Evaluasi yang Digunakan

| Metrik | Keterangan |
|---|---|
| **MAE** (Mean Absolute Error) | Rata-rata selisih absolut antara prediksi dan aktual (dalam Rupiah) |
| **RMSE** (Root Mean Square Error) | Akar kuadrat rata-rata kuadrat selisih — sensitif terhadap outlier |
| **MAPE** (Mean Absolute Percentage Error) | Persentase rata-rata kesalahan — scale-independent, mudah diinterpretasi |

**Kriteria MAPE:**
- < 10% = Sangat Akurat ✅
- 10-20% = Baik/Layak
- 20-50% = Cukup
- > 50% = Tidak Akurat

**Hasil:** Rata-rata MAPE terbaik = **0,86%** (Skenario 6) → **Sangat Akurat**.

---

## 10. Struktur File Proyek

```
Percobaan 2/
├── backend/
│   ├── app.py                          # Flask API Server (492 baris)
│   ├── requirements.txt                # Dependensi Python
│   ├── train_vision_final.py           # Script training final MobileNetV2
│   ├── tune_vision_kfold.py            # Script tuning 5-Fold MobileNetV2
│   ├── balance_dataset.py              # Script undersampling 100/kelas
│   ├── train_final_models.py           # Script training multivariate (percobaan grup)
│   ├── find_best_lstm_params.py        # Script tuning 216 kombinasi
│   ├── models_univariate/              # 8 model LSTM produksi + scaler
│   ├── models_vision/                  # MobileNetV2 model final + evaluasi
│   ├── hasil_tuning_vision_final/      # Rekap hasil tuning K-Fold vision
│   └── Dataset_2017_2026/
│       ├── dataset_harga_komoditas_2017_2026.csv  # Dataset utama
│       ├── dataset_preprocessed.csv                # Dataset ternormalisasi
│       ├── pre-processing.py                       # Script LOCF + normalisasi
│       ├── eksperimen_skripsi.py                   # Script 6 skenario eksperimen
│       ├── train_univariate_final.py               # Script training produksi
│       ├── transpose.py                            # Utilitas transpose data
│       ├── hasil_eksperimen_2/                     # Output 6 skenario + model
│       └── hasil_tuning/                           # Output 216 kombinasi tuning
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # Router (3 halaman)
│   │   ├── components/Navbar.jsx       # Navigasi
│   │   ├── components/Footer.jsx       # Footer
│   │   ├── pages/Home.jsx              # Landing page
│   │   ├── pages/Dashboard.jsx         # Dashboard utama
│   │   └── pages/About.jsx            # Informasi teknis
│   └── package.json                    # React 19, Vite 7, Tailwind 4
├── mobile_app/
│   ├── App.js                          # Seluruh logika mobile (624 baris)
│   ├── app.json                        # Konfigurasi Expo
│   └── package.json                    # RN 0.81, Expo 54
└── hasil_evaluasi_skenario.csv         # Rekapitulasi 6 skenario
```

---

## 11. Status Terkini

| Komponen | Status |
|---|:---:|
| Riset Data & Pra-pemrosesan (LOCF, Normalisasi) | ✅ Selesai |
| Sequential Holdout Tuning (216 Kombinasi) | ✅ Selesai |
| Eksperimen 6 Skenario | ✅ Selesai |
| Pelatihan Model CV (MobileNetV2) | ✅ Selesai |
| Pelatihan 8 Model LSTM Produksi (per-komoditas) | ✅ Selesai |
| API Backend Flask | ✅ Stabil |
| Website Dashboard (React + Vite) | ✅ Selesai |
| Mobile App (React Native + Expo) | ✅ Selesai |
| Laporan Bab 1 (Pendahuluan) | ✅ Selesai |
| Laporan Bab 2 (Tinjauan Pustaka) | ✅ Selesai |
| **Laporan Bab 3 (Kegiatan Pelaksanaan)** | **🔄 Sedang Dikerjakan** |
| Laporan Bab 4 (Analisis dan Pembahasan) | ⬜ Belum |
| Laporan Bab 5 (Kesimpulan) | ⬜ Belum |
