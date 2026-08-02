import { Brain, Network, Cpu, Database } from 'lucide-react';
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
        <p className="text-xl text-gray-600">Sistem Prediksi Harga Komoditas Pangan menggunakan Univariate LSTM</p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-emerald-50 p-8 md:p-12 mb-8">
        <div className="prose prose-emerald max-w-none">
          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <Brain className="text-emerald-600 w-6 h-6" />
            Latar Belakang Proyek Skripsi
          </h2>
          <p className="text-gray-600 leading-relaxed mb-6">
            PanganPredict adalah platform prediksi komoditas pangan yang dibangun sebagai hasil penelitian skripsi. Tujuan utama dari platform ini adalah untuk memprediksi harga bahan pokok di Indonesia dengan tingkat akurasi yang lebih tinggi untuk membantu stabilisasi harga dan pengambilan keputusan.
          </p>
          
          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2 mt-8">
            <Network className="text-emerald-600 w-6 h-6" />
            Arsitektur Univariate LSTM
          </h2>
          <p className="text-gray-600 leading-relaxed mb-6">
            Penelitian skripsi ini membuktikan bahwa pergerakan harga komoditas sangat bersifat autoregresif (bergantung pada sejarah harganya sendiri). Oleh karena itu, sistem ini dirancang menggunakan arsitektur <strong>8 Model Univariate LSTM Independen</strong>. Setiap komoditas memiliki "otak" AI-nya sendiri untuk menghindari kebingungan (noise interference) antar komoditas.
          </p>
          <p className="text-gray-600 leading-relaxed">
            Dilatih dengan dataset ekstensif selama 9 tahun (2017-2026) yang mencakup masa krisis Pandemi COVID-19, model ini menggunakan <strong>Window Size 30 hari dan 64 Unit LSTM</strong>. Konfigurasi ini terbukti memecahkan rekor tingkat kesalahan yang sangat rendah, dengan nilai <strong>MAPE (Mean Absolute Percentage Error) rata-rata hanya 0,86%</strong>.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-100 flex items-start gap-4">
          <div className="bg-emerald-600 p-3 rounded-lg text-white">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 mb-2">Iterative Forecasting</h3>
            <p className="text-sm text-gray-600 leading-relaxed">Sistem mampu meramalkan harga 7 hari ke depan dengan cara berantai (iteratif) berbekal memori 30 hari data masa lalu.</p>
          </div>
        </div>
        <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-100 flex items-start gap-4">
          <div className="bg-emerald-600 p-3 rounded-lg text-white">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 mb-2">Upload Excel Terintegrasi</h3>
            <p className="text-sm text-gray-600 leading-relaxed">Terintegrasi langsung dengan file mentah PIHPS. Pengguna bisa memperbarui data kapan saja tanpa batasan minimal baris.</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default About;
