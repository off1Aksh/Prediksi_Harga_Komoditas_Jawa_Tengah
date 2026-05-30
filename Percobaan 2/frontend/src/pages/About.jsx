import { Brain, Network, Cpu, Database } from 'lucide-react';

const About = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-extrabold text-gray-900 mb-4">Tentang PanganPredict</h1>
        <p className="text-xl text-gray-600">Sistem Prediksi Harga Komoditas Pangan menggunakan Dual-Branch LSTM</p>
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
            Arsitektur Dual-Branch LSTM
          </h2>
          <p className="text-gray-600 leading-relaxed mb-6">
            Pendekatan inovatif yang digunakan dalam penelitian ini adalah penggunaan arsitektur <strong>Dual-Branch Long Short-Term Memory (LSTM)</strong>. Berbeda dengan LSTM konvensional, model ini memisahkan input data ke dalam dua cabang. Cabang pertama berfokus pada komoditas dengan pergerakan harga stabil, sementara cabang kedua menangani komoditas dengan volatilitas harga yang tinggi.
          </p>
          <p className="text-gray-600 leading-relaxed">
            Pembagian arsitektur ini terbukti secara empiris meningkatkan kemampuan model dalam menangkap anomali pasar sekaligus mempertahankan keakuratan tren base-line, menghasilkan performa tinggi yang melampaui pendekatan Single-Branch.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-100 flex items-start gap-4">
          <div className="bg-emerald-600 p-3 rounded-lg text-white">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 mb-2">Pemrosesan Cerdas</h3>
            <p className="text-sm text-gray-600 leading-relaxed">Pemisahan 8 fitur komoditas secara spesifik ke dalam kelompok stabil dan volatil sebelum melalui hidden layer LSTM.</p>
          </div>
        </div>
        <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-100 flex items-start gap-4">
          <div className="bg-emerald-600 p-3 rounded-lg text-white">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 mb-2">Modern Tech Stack</h3>
            <p className="text-sm text-gray-600 leading-relaxed">Integrasi antara React Vite (Frontend) dan Flask/Python (Backend) memastikan latensi minimal pada operasi inferensi machine learning.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
