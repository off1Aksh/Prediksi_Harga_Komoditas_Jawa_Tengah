# 🌾 PanganPredict: Sistem Cerdas Prediksi Harga Komoditas Pangan Jawa Tengah

**PanganPredict** adalah sebuah ekosistem perangkat lunak *full-stack* berbasis *Artificial Intelligence* yang dirancang untuk meramalkan harga 8 komoditas pangan strategis di Provinsi Jawa Tengah selama 7 hari ke depan. Sistem ini menggabungkan model regresi deret waktu (*time-series forecasting*) dengan teknologi *computer vision* untuk memberikan prediksi instan berdasarkan gambar.

Proyek ini dibangun sebagai Tugas Akhir/Skripsi di **Program Studi D4 Teknologi Rekayasa Komputer, Politeknik Negeri Semarang**.

## 🚀 Fitur Utama

* **Prediksi Harga (7 Hari ke Depan):** Menggunakan 8 model *Univariate Long Short-Term Memory* (LSTM) yang dilatih secara independen untuk masing-masing komoditas menggunakan data historis 9 tahun (2017-2026) dari PIHPS Nasional[cite: 14].
* **Deteksi Komoditas Berbasis Gambar:** Mengintegrasikan model *Deep Learning* MobileNetV2 (*Transfer Learning*) untuk mengklasifikasikan jenis komoditas dari jepretan kamera atau unggahan galeri secara otomatis[cite: 14].
* **Pembaruan Data Dinamis:** API yang aman untuk mengunggah dan mem-parsing data Excel PIHPS terbaru, menggabungkannya ke dataset, dan secara otomatis menangani data yang hilang menggunakan metode *Last Observation Carried Forward* (LOCF)[cite: 14].
* **Dashboard Web Analitik:** Antarmuka *Single-Page Application* responsif yang dibangun dengan React 19 dan Tailwind CSS 4, menampilkan visualisasi tren harga (grafik *dual-area*) dan kartu metrik premium yang dianimasikan[cite: 14].
* **Aplikasi Mobile (Cross-Platform):** Aplikasi React Native interaktif yang memungkinkan pengguna memindai gambar komoditas secara langsung melalui kamera *smartphone* untuk mendapatkan prediksi instan[cite: 14].

## 📦 Komoditas yang Didukung

Sistem memantau dan meramalkan harga untuk komoditas berikut[cite: 14]:
1. Bawang Merah
2. Bawang Putih
3. Beras
4. Cabai Merah
5. Cabai Rawit
6. Daging Ayam
7. Minyak Goreng
8. Telur Ayam

## 🏗 Arsitektur Sistem (Tech Stack)

Sistem ini terbagi menjadi tiga komponen utama[cite: 14]:

* **AI & Backend Server:** Python, Flask, TensorFlow / Keras, Scikit-Learn, Pandas, PyJWT.
* **Frontend (Dashboard Web):** React.js (v19), Vite, Tailwind CSS 4, React Router v7, Framer Motion, Recharts.
* **Frontend (Mobile App):** React Native (v0.81), Expo SDK 54, React Native Chart Kit.

## 📊 Performa Model AI

Kombinasi hiperparameter yang digunakan telah diuji melalui pendekatan *Sequential Holdout Tuning* (216 kombinasi) dan eksperimen skenario[cite: 14].

* **Performa LSTM (Rata-rata):** RMSE = Rp591,90 | MAPE = 0,86% (Sangat Akurat)[cite: 14].
* **Performa MobileNetV2:** Akurasi = 95,48% | F1-Score = 95,60%[cite: 14].

---

## 🛠 Panduan Instalasi dan Penggunaan

Repositori ini berisi keseluruhan *source code* (Backend, Frontend, dan Mobile App). Ikuti langkah-langkah di bawah ini untuk menjalankan proyek secara lokal.

### 1. Persiapan Awal
Kloning repositori ini ke komputer Anda:
```bash
git clone [https://github.com/off1Aksh/Prediksi_Harga_Komoditas_Jawa_Tengah.git](https://github.com/off1Aksh/Prediksi_Harga_Komoditas_Jawa_Tengah.git)
cd Prediksi_Harga_Komoditas_Jawa_Tengah/Percobaan\ 2
```

### 2. Konfigurasi Backend (Flask API)
Backend memerlukan Python terinstal dan meng-host model AI serta database CSV[cite: 14].

1. Masuk ke direktori backend:
   ```bash
   cd backend
   ```
2. Buat dan aktifkan *virtual environment* (Sangat disarankan):
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Mac/Linux:
   source venv/bin/activate
   ```
3. Instal semua dependensi Python:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan server Flask:
   ```bash
   python app.py
   ```
   *Server akan berjalan secara default di `http://127.0.0.1:5000` (atau IP lokal Anda).*

### 3. Konfigurasi Frontend (Web Dashboard)
Dashboard web dibangun dengan Vite[cite: 14].

1. Buka terminal baru dan masuk ke direktori frontend:
   ```bash
   cd frontend
   ```
2. Instal dependensi NPM:
   ```bash
   npm install
   ```
3. Jalankan *development server*:
   ```bash
   npm run dev
   ```
   *Aplikasi web dapat diakses melalui URL localhost yang muncul di terminal (biasanya `http://localhost:5173`).*

### 4. Konfigurasi Mobile App (React Native/Expo)
Aplikasi seluler memerlukan aplikasi pendamping Expo Go di ponsel Anda[cite: 14].

1. Buka terminal baru dan masuk ke direktori mobile app:
   ```bash
   cd mobile_app
   ```
2. Instal dependensi NPM:
   ```bash
   npm install
   ```
3. Pastikan konfigurasi IP di file `App.js` telah menunjuk ke IP lokal komputer Anda tempat Backend berjalan (bukan `localhost`)[cite: 14].
4. Mulai server Expo:
   ```bash
   npx expo start
   ```
5. Scan kode QR yang muncul menggunakan aplikasi **Expo Go** di perangkat Android/iOS yang terhubung ke jaringan Wi-Fi yang sama dengan komputer Anda.
