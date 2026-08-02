# Struktur dan Informasi untuk Bab 3 Laporan Skripsi

Berdasarkan laporan skripsi Anda (Bab 1 dan Bab 2) serta struktur referensi dari skripsi Rio, berikut adalah informasi detail proyek Anda yang disusun dan disesuaikan untuk menjadi kerangka **BAB III: KEGIATAN PELAKSANAAN**. Anda dapat memberikan dokumen ini ke AI lain untuk membantu menyusun narasi lengkap Bab 3 Anda.

---

## BAB III KEGIATAN PELAKSANAAN

### 3.1. Metode Pelaksanaan
Pada bagian ini, Anda akan menjelaskan bahwa pengembangan sistem menggunakan metode **Waterfall**.
- **Tahapan Waterfall yang perlu dijelaskan:** 
  1. **Analisis Kebutuhan:** Mengumpulkan kebutuhan untuk prediksi harga (data PIHPS) dan pengembangan aplikasi.
  2. **Desain:** Merancang arsitektur model LSTM Univariate dan desain antarmuka dashboard React JS / React Native.
  3. **Implementasi:** Melakukan pra-pemrosesan data, pelatihan model LSTM, dan penulisan kode web/mobile.
  4. **Pengujian:** Menguji performa model (RMSE, MAE, MAPE) serta menguji fungsionalitas web.
  5. **Penerapan (Deployment) & Pemeliharaan:** Menerapkan model ke server (Flask) dan aplikasi ke pengguna.

### 3.2. Gambaran Umum Sistem
Menjelaskan secara singkat alur sistem prediksi yang dibangun:
- Data historis harian dari PIHPS Nasional dikumpulkan.
- Data diproses dan digunakan untuk melatih **Model LSTM Univariate**.
- Model yang telah dilatih disimpan dan diintegrasikan ke dalam backend **Flask**.
- Frontend berbasis **React JS** (untuk Web) dan **React Native/Expo** (untuk Mobile) mengambil data prediksi dari Flask API dan menampilkannya kepada pengguna dalam bentuk grafik/dashboard interaktif.

### 3.3. Analisis Kebutuhan
#### 3.3.1. Analisis Kebutuhan Data
- **Sumber Data:** Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional.
- **Periode Data:** Tahun 2017 hingga 2026.
- **Wilayah:** Provinsi Jawa Tengah.
- **Komoditas (8 jenis):** Beras, Daging Ayam, Telur Ayam, Bawang Merah, Bawang Putih, Cabai Merah, Cabai Rawit, dan Minyak Goreng.
#### 3.3.2. Analisis Kebutuhan Fungsional
- Pengguna dapat melihat data historis harga pangan.
- Pengguna dapat melihat hasil prediksi harga komoditas pangan untuk 7 hingga 14 hari ke depan.
- Sistem menampilkan visualisasi berupa grafik interaktif pergerakan harga komoditas.
#### 3.3.3. Analisis Kebutuhan Non Fungsional
- **Perangkat Keras (Hardware):** Spesifikasi PC/Laptop yang digunakan untuk _training_ model LSTM (misal: prosesor, RAM, GPU jika ada).
- **Perangkat Lunak (Software):** Python, TensorFlow/Keras, Flask (Backend), Node.js, React JS, React Native, Expo, serta teks editor seperti Visual Studio Code.

### 3.4. Desain Perancangan Sistem
Bagian ini berisi diagram dan perancangan teknis proyek Anda.
#### 3.4.1. Flowchart Sistem
Gambarkan alur kerja mulai dari pengguna membuka aplikasi, memilih komoditas, hingga sistem menampilkan hasil prediksi harga.
#### 3.4.2. Use Case Diagram
Tentukan aktor yang berinteraksi. Umumnya terdapat **Pengguna (Masyarakat/Petani/Pemerintah)** yang melihat informasi prediksi, dan **Sistem/Admin** yang mengelola pembaruan model data harga.
#### 3.4.3. Activity Diagram
Diagram yang merinci langkah-langkah aktivitas dari Use Case di atas.
#### 3.4.4. Pembuatan Antarmuka Aplikasi
Rancangan UI/UX (Wireframe/Mockup) untuk halaman Web (React JS) dan halaman Mobile (React Native) yang menampilkan dasbor harga.
#### 3.4.5. Persiapan Data (Data Pre-processing)
Sesuai landasan teori Anda, jelaskan langkah-langkah:
- **Normalisasi:** Menggunakan **MinMaxScaler** untuk mengubah rentang data ke 0-1.
- **Sliding Window:** Mengubah data time-series menjadi supervised learning dengan variasi *window size* **7 hari, 14 hari, dan 30 hari**.
- **Data Splitting:** Pembagian data latih dan uji menggunakan teknik **Sequential Holdout / Time-Series Split** agar urutan waktu tetap terjaga.
#### 3.4.6. Pembuatan Model
Menjelaskan arsitektur **LSTM Univariate** yang memodelkan masing-masing dari 8 komoditas secara terpisah. Sebutkan eksperimen konfigurasi model menggunakan **32 unit dan 64 unit LSTM**.
#### 3.4.7. Penulisan Kode Aplikasi
Menjelaskan proses pengkodean menggunakan Python untuk model Machine Learning dan Flask API, serta Javascript untuk React JS (Web) dan React Native (Mobile).

### 3.5. Rancangan Pengujian Sistem
#### 3.5.1. Pengujian Performa Algoritma
Menjelaskan bahwa model dievaluasi berdasarkan nilai error terkecil dari variasi _sliding window_ dan jumlah unit LSTM. Metrik evaluasi yang digunakan adalah:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**
- **MAPE (Mean Absolute Percentage Error)**
#### 3.5.2. Pengujian Fungsionalitas Aplikasi
Pengujian *Black-box* untuk mengecek apakah fitur-fitur pada antarmuka web dan mobile berfungsi sesuai harapan (misal: pengujian perpindahan halaman, pemfilteran komoditas, responsibilitas grafik).
#### 3.5.3. Pengujian Kepuasan Pengguna *(Opsional)*
Jika Anda berencana menyebarkan kuesioner kepada responden (misal: petani atau masyarakat) untuk mengukur tingkat kepuasan dan kemudahan penggunaan aplikasi berbasis web dan mobile Anda, maka bagian ini dapat disertakan menggunakan kuesioner skala likert (seperti di referensi).

---
**Catatan untuk AI Penyusun Bab 3:** 
Gunakan kerangka dan rincian data (dataset, algoritma, framework, dan metrik) di atas untuk menyusun narasi lengkap Bab 3 menggunakan tata bahasa akademik yang sesuai dengan kaidah penulisan skripsi.
