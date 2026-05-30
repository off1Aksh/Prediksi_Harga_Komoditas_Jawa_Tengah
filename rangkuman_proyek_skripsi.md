# 📋 Rangkuman Lengkap Proyek Skripsi
## Sistem Prediksi Harga Komoditas Pangan — Univariate LSTM

**Nama Mahasiswa:** Akasha Bin Ali  
**Lokasi Proyek:** `d:\KULIAH\SKRIPSI\Percobaan 2\`  
**Tanggal Update Terakhir:** 29 Mei 2026

---

## 1. Deskripsi Umum Proyek

Proyek skripsi ini membangun **Sistem Prediksi Harga 8 Komoditas Pangan** di Indonesia menggunakan algoritma **Long Short-Term Memory (LSTM)** dengan pendekatan **Univariate**. Sistem ini melatih 8 "otak" AI yang terpisah untuk masing-masing komoditas guna menghindari *noise interference*. Sistem ini diintegrasikan ke dalam sebuah **aplikasi web full-stack** (Flask & React) untuk visualisasi yang modern dan dinamis.

### 8 Komoditas yang Diprediksi:
Bawang Merah, Bawang Putih, Beras, Cabai Merah, Cabai Rawit, Daging Ayam, Minyak Goreng, Telur Ayam.

---

## 2. Pembaruan Signifikan (Pivot Riset)

Dalam proses penelitiannya, riset ini melakukan dua *pivot* besar berdasarkan temuan data:
1. **Perluasan Dataset:** Dari 4 tahun (2021-2026) menjadi **9 tahun (2017-2026)** untuk menguji model pada data yang mencakup krisis COVID-19 dan pemulihan ekonomi (Total: 3.195 baris).
2. **Perubahan Arsitektur:** Beralih dari *Multivariate Dual-Branch* ke **8 Model Univariate Terpisah**. Hasil eksperimen membuktikan bahwa harga komoditas sangat bersifat autoregresif, dan Univariate memenangkan akurasi di 7 dari 8 komoditas.

---

## 3. Arsitektur Model Univariate LSTM

Karena pendekatannya Univariate, sistem ini membuat 8 model independen. Masing-masing model memiliki arsitektur yang sama namun dilatih secara spesifik dengan data historis komoditasnya sendiri:
- **Input:** `(Window Size, 1)`
- **Layer 1:** LSTM (32 atau 64 unit) dengan aktivasi `tanh`.
- **Layer 2:** Dense (1 unit) untuk regresi harga.
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Scaler:** MinMaxScaler global (`scaler_master.gz`) di-*fit* pada 100% dataset mentah untuk menghindari *double-scaling*.

---

## 4. Eksperimen Laboratorium (Skripsi)

Untuk memenuhi janji di Proposal Skripsi, dilakukan pengujian *Hyperparameter Tuning* dengan 6 skenario kombinasi *Window Size* (7, 14, 30 hari) dan jumlah Unit LSTM (32, 64 unit). 

Data dibagi secara kronologis: **80% Training dan 20% Testing**. 
Pelatihan menggunakan **Early Stopping** (Patience: 50, Start From Epoch: 50, Max Epochs: 1000) untuk mencari bobot terbaik yang menghindari "Jebakan Epoch 1".

### Hasil Evaluasi (Rata-rata dari 8 Komoditas):
| Skenario | Window Size | LSTM Units | Rata-rata RMSE (Rp) | Rata-rata MAPE (%) |
|----------|:-----------:|:----------:|:-------------------:|:------------------:|
| Skenario 1 | 7 | 32 | 636.24 | 1.02% |
| Skenario 2 | 7 | 64 | 637.97 | 1.05% |
| Skenario 3 | 14 | 32 | 618.50 | 1.01% |
| Skenario 4 | 14 | 64 | 602.56 | 1.00% |
| Skenario 5 | 30 | 32 | 593.04 | 0.88% |
| **Skenario 6** | **30** | **64** | **591.90** | **0.86%** 🏆 |

**Kesimpulan Ilmiah:** Skenario 6 keluar sebagai juara dengan MAPE sangat rendah (0,86%). Ini membuktikan bahwa LSTM bekerja paling maksimal jika diberikan konteks sejarah harga sebulan penuh (Window 30) dipadukan dengan jaringan saraf yang lebih kompleks (LSTM 64).

---

## 5. Implementasi ke Sistem Produksi (Website)

Setelah mendapatkan konfigurasi terbaik (W30, L64), model *Production* dilatih menggunakan skrip `train_univariate_final.py` dengan menggunakan **100% dataset (tanpa validasi split)** hingga 1500 Epoch. 
Tujuannya agar model menyerap seluruh memori masa lalu hingga hari terakhir sebelum di-deploy ke dunia nyata. 

### Backend (Flask - `app.py`)
- Meload 8 file `.keras` dan 1 `scaler_master.gz` dari folder `models_univariate`.
- Menerima 30 hari data dari React.
- Melakukan **Iterative Forecasting** secara *looping* untuk menghasilkan tebakan 7 hari ke depan.

### Frontend (React/Vite)
- Menampilkan grafik gabungan (historis + 7 hari prediksi) menggunakan *Recharts*.
- Desain *glassmorphism* modern dengan *TailwindCSS*.

---

## 6. Status Terkini
✅ **Riset Data:** Selesai (Dataset 2017-2026 bersih dan ternormalisasi).
✅ **Eksperimen Model:** Selesai (Skenario Juara: W30, L64).
✅ **Pembangunan Sistem:** Selesai (API berjalan, UI interaktif dan responsif).

**Fokus Berikutnya:** Penulisan Laporan Bab 4 (Hasil dan Pembahasan) ke dalam Skripsi, berbekal seluruh grafik dan tabel CSV dari `hasil_eksperimen_2`.
