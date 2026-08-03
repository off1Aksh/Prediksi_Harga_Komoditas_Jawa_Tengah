import { LineChart, Camera, Brain, Network, Database, Cpu, User } from 'lucide-react';
import { motion } from 'framer-motion';

const About = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="max-w-4xl mx-auto px-4 py-16 sm:px-6 lg:px-8"
    >
      <div className="text-center mb-16">
        <h1 className="text-4xl font-extrabold text-gray-900 mb-4">Tentang PanganPredict</h1>
        <p className="text-xl text-gray-600">Sistem Cerdas Pangan Berbasis AI (Computer Vision & Time Series Regression)</p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-emerald-50 p-8 md:p-12 mb-8 relative overflow-hidden">
        {/* Subtle decorative background */}
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-emerald-50 rounded-full blur-3xl opacity-50 pointer-events-none"></div>
        
        <div className="prose prose-emerald max-w-none relative z-10">
          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <User className="text-emerald-600 w-6 h-6" />
            Profil Pengembang
          </h2>
          <p className="text-gray-600 leading-relaxed mb-6">
            Proyek skripsi ini dikembangkan oleh <strong>Akasha Bin Ali (NIM: 4.33.22.0.01)</strong>, mahasiswa program studi S.Tr Teknologi Rekayasa Komputer di <strong>Politeknik Negeri Semarang</strong>. Proyek ini bertujuan untuk mendeteksi jenis komoditas pangan dari gambar dan meramalkan harga komoditas di Provinsi Jawa Tengah untuk 7 hari ke depan.
          </p>

          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2 mt-8">
            <Brain className="text-emerald-600 w-6 h-6" />
            Dua Pilar Kecerdasan Buatan (AI)
          </h2>
          <p className="text-gray-600 leading-relaxed mb-6">
            PanganPredict menggunakan pendekatan <em>Full-Stack AI</em> dengan menggabungkan dua model utama untuk menghadirkan pengalaman pengguna yang optimal, baik dari aplikasi mobile maupun dashboard web:
          </p>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 my-8 not-prose">
            <div className="bg-emerald-50/50 rounded-xl p-6 border border-emerald-100">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-emerald-600 p-2 rounded-lg text-white">
                  <Camera className="w-5 h-5" />
                </div>
                <h3 className="font-bold text-gray-900 m-0">MobileNetV2 (Vision)</h3>
              </div>
              <p className="text-sm text-gray-600 m-0 leading-relaxed">
                Model <em>Transfer Learning</em> untuk mendeteksi 8 jenis komoditas dari gambar. Dilatih menggunakan <em>Stratified 5-Fold Cross-Validation</em> dengan tingkat akurasi mencapai <strong>95.48%</strong>.
              </p>
            </div>
            
            <div className="bg-emerald-50/50 rounded-xl p-6 border border-emerald-100">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-emerald-600 p-2 rounded-lg text-white">
                  <LineChart className="w-5 h-5" />
                </div>
                <h3 className="font-bold text-gray-900 m-0">Univariate LSTM (Time Series)</h3>
              </div>
              <p className="text-sm text-gray-600 m-0 leading-relaxed">
                Terdiri dari 8 model independen untuk masing-masing komoditas. Berdasarkan eksperimen, skenario terbaik mencatatkan nilai rata-rata kesalahan <strong>MAPE sangat akurat (0,86%)</strong>.
              </p>
            </div>
          </div>

          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2 mt-8">
            <Network className="text-emerald-600 w-6 h-6" />
            Arsitektur & Hyperparameter Tuning
          </h2>
          <p className="text-gray-600 leading-relaxed mb-6">
            Untuk menghindari <em>noise interference</em> antar komoditas yang ditemukan pada pendekatan multivariat, PanganPredict menggunakan <strong>8 Model Univariate LSTM Independen</strong>. 
            Setiap model telah melewati proses <em>Sequential Holdout Tuning</em> dengan 216 kombinasi untuk menemukan konfigurasi <em>Window Size</em> (7, 14, 30) dan <em>LSTM Units</em> (32, 64, 128) yang paling optimal per komoditas.
          </p>
          <p className="text-gray-600 leading-relaxed mb-6">
            Data historis diambil dari Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional selama 9 tahun (2017-2026), yang kemudian melewati pra-pemrosesan menggunakan metode <em>Last Observation Carried Forward (LOCF)</em> dan normalisasi Min-Max.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-100 flex items-start gap-4">
          <div className="bg-emerald-600 p-3 rounded-lg text-white shrink-0">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 mb-2">Iterative Forecasting</h3>
            <p className="text-sm text-gray-600 leading-relaxed">Sistem mampu meramalkan harga 7 hari ke depan secara berantai (iteratif) menggunakan pendekatan <em>autoregressive</em> berbekal memori masa lalu.</p>
          </div>
        </div>
        <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-100 flex items-start gap-4">
          <div className="bg-emerald-600 p-3 rounded-lg text-white shrink-0">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 mb-2">Upload Data Dinamis</h3>
            <p className="text-sm text-gray-600 leading-relaxed">Terintegrasi langsung dengan file Excel mentah PIHPS. Pengguna (Admin) bisa memperbarui data dan sistem akan memproses ekstraksi serta LOCF secara otomatis.</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default About;
