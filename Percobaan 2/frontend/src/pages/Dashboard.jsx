import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader2, AlertCircle } from 'lucide-react';

const COMMODITIES = [
  "Bawang Merah",
  "Bawang Putih",
  "Beras",
  "Cabai Merah",
  "Cabai Rawit",
  "Daging Ayam",
  "Minyak Goreng",
  "Telur Ayam"
];

const COMMODITY_IMAGES = {
  "Bawang Merah": "/commodities/bawang_merah.jpg",
  "Bawang Putih": "/commodities/bawang_putih.jpg",
  "Beras": "/commodities/beras.jpg",
  "Cabai Merah": "/commodities/cabai_merah.jpg",
  "Cabai Rawit": "/commodities/cabai_rawit.jpg",
  "Daging Ayam": "/commodities/daging_ayam.jpg",
  "Minyak Goreng": "/commodities/minyak_goreng.jpg",
  "Telur Ayam": "/commodities/telur_ayam.jpg"
};



const Dashboard = () => {
  const [historicalData, setHistoricalData] = useState([]);
  const [historicalDates, setHistoricalDates] = useState([]);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCommodity, setSelectedCommodity] = useState(0);
  
  // State untuk Auth
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchData();
  }, []);

  const handleUploadExcel = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);
      const token = localStorage.getItem('adminToken');
      const response = await axios.post('/api/upload-excel', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        }
      });
      alert(response.data.message);
      await fetchData(); // Refresh data setelah upload
    } catch (err) {
      if (err.response?.status === 401) {
        localStorage.removeItem('adminToken');
        alert("Sesi Anda telah habis atau Anda tidak memiliki akses. Silakan login kembali.");
        setShowLoginModal(true);
      } else {
        alert(err.response?.data?.error || 'Gagal mengupload file Excel');
      }
    } finally {
      setLoading(false);
      event.target.value = null; 
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    try {
      const response = await axios.post('/api/login', { username, password });
      localStorage.setItem('adminToken', response.data.token);
      setShowLoginModal(false);
      setUsername('');
      setPassword('');
      // Langsung buka file picker setelah berhasil login
      fileInputRef.current?.click();
    } catch (err) {
      setLoginError(err.response?.data?.error || 'Gagal login. Username atau password salah.');
    }
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // 1. Ambil data aktual (Rupiah) dari backend
      const resLatest = await fetch('/api/latest-data');
      if (!resLatest.ok) throw new Error('Gagal mengambil data historis (latest-data)');
      const dataLatest = await resLatest.json();

      setHistoricalData(dataLatest.historical_data);
      setHistoricalDates(dataLatest.dates || []);

      // 2. Minta Prediksi (backend akan menggunakan data preprocessed internal)
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error('Gagal mengambil data prediksi dari server');
      }

      const result = await response.json();
      setForecastData(result.forecast);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Format angka ke Rupiah Indonesia.
   * Contoh: 40000 → "Rp 40.000"
   */
  const formatRupiah = (number) => {
    if (number === undefined || number === null) return '-';
    return new Intl.NumberFormat('id-ID', {
      style: 'currency',
      currency: 'IDR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(number);
  };

  /**
   * Format angka pendek untuk Y-axis (misalnya: 40.000)
   */
  const formatYAxis = (value) => {
    if (value === undefined || value === null) return '';
    return new Intl.NumberFormat('id-ID').format(value);
  };

  // Persiapkan data untuk chart berdasarkan komoditas yang dipilih
  const getChartData = () => {
    if (!historicalData.length) return [];

    const chartData = [];

    // Data Historis (30 hari)
    for (let i = 0; i < 30; i++) {
      const isToday = i === 29;
      // Gunakan tanggal aktual jika tersedia
      let label;
      if (isToday) {
        label = 'Hari Ini';
      } else if (historicalDates[i]) {
        const d = new Date(historicalDates[i]);
        label = `${d.getDate()}/${d.getMonth() + 1}`;
      } else {
        label = `H-${29 - i}`;
      }

      chartData.push({
        day: label,
        historis: historicalData[i][selectedCommodity],
        prediksi: null,
      });
    }

    // Untuk menghubungkan garis antara historis dan prediksi di chart
    if (forecastData && forecastData.length > 0) {
      chartData[29].prediksi = chartData[29].historis;

      // Data Prediksi (7 hari)
      const lastDateStr = historicalDates[29];
      const lastDateObj = lastDateStr ? new Date(lastDateStr) : new Date();

      for (let i = 0; i < 7; i++) {
        const nextDate = new Date(lastDateObj);
        nextDate.setDate(lastDateObj.getDate() + i + 1);
        
        const day = nextDate.getDate().toString().padStart(2, '0');
        const month = (nextDate.getMonth() + 1).toString().padStart(2, '0');
        const nextDateLabel = `${day}/${month}`;
        
        const fullDateStr = nextDate.toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' });

        chartData.push({
          day: nextDateLabel,
          fullDate: fullDateStr,
          historis: null,
          prediksi: forecastData[i][selectedCommodity],
        });
      }
    }

    return chartData;
  };

  const chartData = getChartData();

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="min-h-screen bg-slate-50"
    >
      <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard Analisis</h1>
          <p className="text-gray-600 mt-1">Pemantauan harga komoditas dan prediksi 7 hari ke depan</p>
        </div>
        <div className="mt-4 md:mt-0 flex gap-3">
          <input
            type="file"
            accept=".xls,.xlsx"
            className="hidden"
            ref={fileInputRef}
            onChange={handleUploadExcel}
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={fetchData}
            disabled={loading}
            className="px-4 py-2 bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh Grafik'}
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              const token = localStorage.getItem('adminToken');
              if (token) {
                fileInputRef.current?.click();
              } else {
                setShowLoginModal(true);
              }
            }}
            disabled={loading}
            className="px-4 py-2 bg-emerald-100 text-emerald-700 hover:bg-emerald-200 rounded-lg font-medium transition-colors disabled:opacity-50 shadow-sm"
          >
            {loading ? 'Memproses File...' : 'Upload Excel (Perbarui)'}
          </motion.button>
        </div>
      </div>

      {error && (
        <div className="mb-8 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <p>{error}</p>
        </div>
      )}

      {loading ? (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="flex flex-col items-center justify-center py-20 min-h-[400px]"
        >
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mb-4" />
          <p className="text-gray-600">Model sedang memproses prediksi tingkat lanjut...</p>
        </motion.div>
      ) : (
        <>
          {/* Kartu Ringkasan Harga (Harga Terakhir dari history) */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {COMMODITIES.map((name, index) => {
              const currentPrice = historicalData[29]?.[index] || 0;
              const isSelected = selectedCommodity === index;
              return (
                <motion.div
                  key={name}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => setSelectedCommodity(index)}
                  className={`cursor-pointer rounded-2xl p-5 border transition-all relative overflow-hidden ${isSelected
                    ? 'bg-gradient-to-br from-cyan-50 to-blue-50 border-cyan-200 shadow-[0_0_15px_rgba(6,182,212,0.3)]'
                    : 'bg-white border-gray-100 shadow-sm hover:border-emerald-200 hover:shadow-md'
                    }`}
                >
                  {/* Gambar Latar Belakang Transparan (Watermark) */}
                  {COMMODITY_IMAGES[name] ? (
                    <img
                      src={COMMODITY_IMAGES[name]}
                      alt={name}
                      className="absolute -right-4 -bottom-4 w-28 h-28 object-cover opacity-20 mix-blend-multiply rotate-[-15deg] pointer-events-none grayscale-[20%]"
                    />
                  ) : null}

                  <div className="flex justify-between items-start relative z-10">
                    <p className={`text-base font-bold mb-1 truncate ${isSelected ? 'text-cyan-900' : 'text-gray-700'}`}>{name}</p>
                  </div>
                  <p className={`text-2xl font-bold mt-2 text-gray-900 relative z-10`}>
                    {formatRupiah(currentPrice)}
                  </p>
                  <p className="text-xs text-gray-400 mt-2 relative z-10">Update hari ini</p>
                </motion.div>
              );
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Area Grafik */}
            <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-emerald-50">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  Grafik Harga <span className="px-2 py-1 bg-cyan-100 text-cyan-800 text-sm rounded-md ml-2">{COMMODITIES[selectedCommodity]}</span>
                </h3>
                <select
                  className="bg-gray-50 border border-gray-200 text-gray-900 text-sm rounded-lg focus:ring-emerald-500 focus:border-emerald-500 block p-2"
                  value={selectedCommodity}
                  onChange={(e) => setSelectedCommodity(Number(e.target.value))}
                >
                  {COMMODITIES.map((name, idx) => (
                    <option key={idx} value={idx}>{name}</option>
                  ))}
                </select>
              </div>

              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                    <defs>
                      <linearGradient id="colorHistoris" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.0}/>
                      </linearGradient>
                      <linearGradient id="colorPrediksi" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                    <XAxis
                      dataKey="day"
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      tickLine={false}
                      axisLine={{ stroke: '#e5e7eb' }}
                    />
                    <YAxis
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={formatYAxis}
                      width={80}
                    />
                    <Tooltip
                      formatter={(value) => formatRupiah(value)}
                      labelStyle={{ color: '#374151', fontWeight: 'bold' }}
                      contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)' }}
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    <Area
                      type="monotone"
                      dataKey="historis"
                      name="Harga Historis"
                      stroke="#10b981"
                      strokeWidth={3}
                      fillOpacity={1}
                      fill="url(#colorHistoris)"
                      dot={{ r: 4, strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6, strokeWidth: 0, fill: '#059669' }}
                      connectNulls
                    />
                    <Area
                      type="monotone"
                      dataKey="prediksi"
                      name="Prediksi (7 Hari)"
                      stroke="#f59e0b"
                      strokeWidth={3}
                      strokeDasharray="7 5"
                      fillOpacity={1}
                      fill="url(#colorPrediksi)"
                      dot={{ r: 4, strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6, strokeWidth: 0, fill: '#d97706' }}
                      connectNulls
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Area Tabel Detail Prediksi */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-emerald-50 max-h-[440px] overflow-auto">
              <h3 className="text-lg font-bold text-gray-900 mb-6">Prediksi 7 Hari ke Depan</h3>
              {forecastData && forecastData.length > 0 ? (
                <div className="space-y-4">
                  {forecastData.map((dayPred, index) => {
                    const predValue = dayPred[selectedCommodity];
                    const todayValue = historicalData[29]?.[selectedCommodity];
                    const diff = predValue - todayValue;
                    const isUp = diff > 0;

                    return (
                      <div key={index} className="flex justify-between items-center p-3 hover:bg-slate-50 rounded-lg transition-colors border border-transparent hover:border-slate-200">
                        <span className="font-medium text-gray-700">{chartData[30 + index]?.fullDate || `H+${index + 1}`}</span>
                        <div className="flex flex-col items-end">
                          <p className="font-bold text-gray-900">{formatRupiah(predValue)}</p>
                          <div className={`mt-2 flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold ${isUp ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'}`}>
                            {isUp ? <span>↗</span> : <span>↓</span>}
                            <span>{isUp ? '+' : '-'}{formatRupiah(Math.abs(diff))} vs Hari Ini</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-10 text-gray-500">
                  Data prediksi belum tersedia
                </div>
              )}
            </div>
          </div>
        </>
      )}
      </div>

      <AnimatePresence>
        {showLoginModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }} 
              transition={{ duration: 0.2 }}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
              onClick={() => setShowLoginModal(false)}
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }} 
              animate={{ opacity: 1, scale: 1, y: 0 }} 
              exit={{ opacity: 0, scale: 0.9, y: 20 }} 
              transition={{ type: "spring", damping: 20, stiffness: 300 }}
              className="relative bg-white rounded-2xl shadow-xl w-full max-w-md p-6 sm:p-8 overflow-hidden border border-slate-100"
            >
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Login Admin</h3>
              <p className="text-gray-500 mb-6">Otentikasi diperlukan untuk memperbarui dataset PIHPS Nasional.</p>
              
              {loginError && (
                <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-4 border border-red-100">
                  {loginError}
                </div>
              )}

              <form onSubmit={handleLogin} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                  <input type="text" value={username} onChange={e => setUsername(e.target.value)} className="w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all" placeholder="Masukkan username" required />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                  <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all" placeholder="••••••••" required />
                </div>
                <div className="flex gap-3 pt-2">
                  <button type="button" onClick={() => setShowLoginModal(false)} className="flex-1 px-4 py-2.5 bg-gray-100 text-gray-700 rounded-xl font-semibold hover:bg-gray-200 transition-colors">Batal</button>
                  <button type="submit" className="flex-1 px-4 py-2.5 bg-emerald-600 text-white rounded-xl font-semibold hover:bg-emerald-700 shadow-sm shadow-emerald-200 transition-colors">Masuk</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

    </motion.div>
  );
};

export default Dashboard;
