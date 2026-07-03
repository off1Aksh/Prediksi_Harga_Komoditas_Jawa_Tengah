import { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
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



const Dashboard = () => {
  const [historicalData, setHistoricalData] = useState([]);
  const [historicalDates, setHistoricalDates] = useState([]);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCommodity, setSelectedCommodity] = useState(0);
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchData();
  }, []);

  const handleUploadExcel = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:5000/api/upload-excel', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Gagal mengunggah file Excel');
      }

      event.target.value = null; // reset input
      
      // Ambil grafik terbaru setelah data berhasil disatukan di backend
      await fetchData(); 
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // 1. Ambil data aktual (Rupiah) dari backend
      const resLatest = await fetch('http://localhost:5000/api/latest-data');
      if (!resLatest.ok) throw new Error('Gagal mengambil data historis (latest-data)');
      const dataLatest = await resLatest.json();

      setHistoricalData(dataLatest.historical_data);
      setHistoricalDates(dataLatest.dates || []);

      // 2. Minta Prediksi (backend akan menggunakan data preprocessed internal)
      const response = await fetch('http://localhost:5000/api/predict', {
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
          <button
            onClick={fetchData}
            disabled={loading}
            className="px-4 py-2 bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh Grafik'}
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="px-4 py-2 bg-emerald-100 text-emerald-700 hover:bg-emerald-200 rounded-lg font-medium transition-colors disabled:opacity-50 shadow-sm"
          >
            {loading ? 'Memproses File...' : 'Upload Excel (Perbarui)'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-8 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <p>{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 min-h-[400px]">
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mb-4" />
          <p className="text-gray-600">Model sedang memproses prediksi tingkat lanjut...</p>
        </div>
      ) : (
        <>
          {/* Kartu Ringkasan Harga (Harga Terakhir dari history) */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {COMMODITIES.map((name, index) => {
              const currentPrice = historicalData[29]?.[index] || 0;
              const isSelected = selectedCommodity === index;
              return (
                <div
                  key={name}
                  onClick={() => setSelectedCommodity(index)}
                  className={`cursor-pointer rounded-xl p-4 border transition-all ${isSelected
                    ? 'bg-emerald-50 border-emerald-500 shadow-md transform scale-[1.02]'
                    : 'bg-white border-gray-100 shadow-sm hover:border-emerald-300 hover:shadow'
                    }`}
                >
                  <p className="text-sm text-gray-500 font-medium mb-1 truncate">{name}</p>
                  <p className={`text-xl font-bold ${isSelected ? 'text-emerald-700' : 'text-gray-900'}`}>
                    {formatRupiah(currentPrice)}
                  </p>
                  <p className="text-xs text-gray-400 mt-2">Update hari ini</p>
                </div>
              );
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Area Grafik */}
            <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-emerald-50">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  Grafik Harga <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-sm rounded-md ml-2">{COMMODITIES[selectedCommodity]}</span>
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
                  <LineChart data={chartData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
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
                    <Line
                      type="monotone"
                      dataKey="historis"
                      name="Harga Historis"
                      stroke="#10b981"
                      strokeWidth={3}
                      dot={{ r: 4, strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6, strokeWidth: 0, fill: '#059669' }}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="prediksi"
                      name="Prediksi (7 Hari)"
                      stroke="#f59e0b" /* Menggunakan warna amber agar kontras dan mudah dibedakan */
                      strokeWidth={3}
                      strokeDasharray="7 5"
                      dot={{ r: 4, strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6, strokeWidth: 0, fill: '#d97706' }}
                      connectNulls
                    />
                  </LineChart>
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
                      <div key={index} className="flex justify-between items-center p-3 hover:bg-emerald-50 rounded-lg transition-colors border border-transparent hover:border-emerald-100">
                        <span className="font-medium text-gray-700">{chartData[30 + index]?.fullDate || `H+${index + 1}`}</span>
                        <div className="text-right">
                          <p className="font-bold text-gray-900">{formatRupiah(predValue)}</p>
                          <p className={`text-xs font-medium mt-1 ${isUp ? 'text-red-500' : 'text-emerald-500'}`}>
                            {isUp ? '+' : ''}{formatRupiah(diff)} vs Hari Ini
                          </p>
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
  );
};

export default Dashboard;
