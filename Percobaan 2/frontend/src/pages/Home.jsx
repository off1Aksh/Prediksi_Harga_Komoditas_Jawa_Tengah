import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { TrendingUp, Activity, Zap, ArrowRight } from 'lucide-react';

const FeatureCard = ({ icon: Icon, title, description }) => (
  <div className="bg-white p-6 rounded-2xl shadow-sm border border-emerald-50 hover:shadow-md transition-shadow">
    <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center mb-4 text-emerald-600">
      <Icon className="w-6 h-6" />
    </div>
    <h3 className="text-lg font-bold text-gray-900 mb-2">{title}</h3>
    <p className="text-gray-600 leading-relaxed">{description}</p>
  </div>
);

const Home = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col min-h-full"
    >
      {/* Hero Section */}
      <section className="relative pt-20 pb-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-50 to-white -z-10" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-100 text-emerald-700 text-sm font-medium mb-8">
            <span className="flex w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
            Univariate LSTM Model
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold text-gray-900 tracking-tight mb-8">
            <span className="block">Prediksi Harga Komoditas</span>
            <span className="block text-emerald-600">Lebih Akurat & Cerdas</span>
          </h1>
          <p className="max-w-2xl mx-auto text-xl text-gray-600 mb-10 leading-relaxed">
            Platform prediktif PanganPredict menggunakan arsitektur deep learning mutakhir untuk memberikan wawasan harga pangan di masa depan dengan presisi tinggi.
          </p>
          <div className="flex justify-center gap-4 text-center">
            <Link
              to="/dashboard"
              className="inline-flex items-center justify-center px-8 py-4 text-base font-bold text-white bg-emerald-600 hover:bg-emerald-700 rounded-full shadow-lg hover:shadow-emerald-500/30 transition-all gap-2"
            >
              Mulai Analisis
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900">Mengapa PanganPredict?</h2>
            <p className="mt-4 text-lg text-gray-600">Fitur unggulan untuk keakuratan prediksi Anda</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard 
              icon={TrendingUp}
              title="Akurasi Tinggi (MAPE 0.86%)"
              description="Menggunakan 8 Model Univariate LSTM independen dengan Window Size 30 dan 64 Unit LSTM untuk akurasi prediksi maksimal."
            />
            <FeatureCard 
              icon={Activity}
              title="Upload Dinamis PIHPS"
              description="Sistem terintegrasi langsung dengan format Excel mentah Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional."
            />
            <FeatureCard 
              icon={Zap}
              title="Dataset Pandemi 9 Tahun"
              description="Dilatih menggunakan dataset ekstensif dari 2017 hingga 2026 yang terbukti tangguh memprediksi di masa krisis COVID-19."
            />
          </div>
        </div>
      </section> */}
    </motion.div>
  );
};

export default Home;
