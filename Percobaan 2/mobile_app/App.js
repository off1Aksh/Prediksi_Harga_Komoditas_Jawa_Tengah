import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ActivityIndicator, ScrollView, Alert, Modal, FlatList, Dimensions } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { LineChart } from 'react-native-chart-kit';
import { Leaf, Menu, X, Home, Calendar, Search } from 'lucide-react-native';

const API_URL_IMAGE = "http://192.168.1.84:5000/api/predict-image";
const API_URL_MANUAL = "http://192.168.1.84:5000/api/predict-single";
const API_URL_HISTORY = "http://192.168.1.84:5000/api/history";

const COMMODITIES = [
  { id: 'bawang_merah', name: 'Bawang Merah' },
  { id: 'bawang_putih', name: 'Bawang Putih' },
  { id: 'beras', name: 'Beras' },
  { id: 'cabai_merah', name: 'Cabai Merah' },
  { id: 'cabai_rawit', name: 'Cabai Rawit' },
  { id: 'daging_ayam', name: 'Daging Ayam' },
  { id: 'minyak_goreng', name: 'Minyak Goreng' },
  { id: 'telur_ayam', name: 'Telur Ayam' },
];

const YEARS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"];
const MONTHS = [
  { id: '1', name: 'Januari' }, { id: '2', name: 'Februari' }, { id: '3', name: 'Maret' },
  { id: '4', name: 'April' }, { id: '5', name: 'Mei' }, { id: '6', name: 'Juni' },
  { id: '7', name: 'Juli' }, { id: '8', name: 'Agustus' }, { id: '9', name: 'September' },
  { id: '10', name: 'Oktober' }, { id: '11', name: 'November' }, { id: '12', name: 'Desember' }
];

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [sidebarVisible, setSidebarVisible] = useState(false);

  // Home states
  const [imageUri, setImageUri] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedCommodity, setSelectedCommodity] = useState(null);

  // History states
  const [histCommodity, setHistCommodity] = useState(null);
  const [histYear, setHistYear] = useState(null);
  const [histMonth, setHistMonth] = useState(null);
  const [histResult, setHistResult] = useState(null);
  const [histLoading, setHistLoading] = useState(false);

  // General Picker Modal
  const [modalVisible, setModalVisible] = useState(false);
  const [pickerType, setPickerType] = useState(null);

  const openPicker = (type) => {
    setPickerType(type);
    setModalVisible(true);
  };

  const openCamera = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (permissionResult.granted === false) {
      Alert.alert("Izin Ditolak", "Anda harus mengizinkan akses kamera.");
      return;
    }
    const pickerResult = await ImagePicker.launchCameraAsync({ allowsEditing: true, aspect: [1, 1], quality: 0.8 });
    if (!pickerResult.canceled) {
      const uri = pickerResult.assets[0].uri;
      setImageUri(uri);
      setSelectedCommodity(null);
      setResult(null);
      analyzeImage(uri);
    }
  };

  const openGallery = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permissionResult.granted === false) {
      Alert.alert("Izin Ditolak", "Anda harus mengizinkan akses galeri.");
      return;
    }
    const pickerResult = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, allowsEditing: true, aspect: [1, 1], quality: 0.8 });
    if (!pickerResult.canceled) {
      const uri = pickerResult.assets[0].uri;
      setImageUri(uri);
      setSelectedCommodity(null);
      setResult(null);
      analyzeImage(uri);
    }
  };

  const analyzeImage = async (uri) => {
    setLoading(true);
    const filename = uri.split('/').pop();
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image`;
    const formData = new FormData();
    formData.append('image', { uri: uri, name: filename, type });

    try {
      const response = await axios.post(API_URL_IMAGE, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResult(response.data);
    } catch (error) {
      Alert.alert("Error Server", error.response?.data?.error || "Gagal memproses gambar.");
    } finally {
      setLoading(false);
    }
  };

  const predictManual = async (komoditasId, komoditasName) => {
    setImageUri(null);
    setSelectedCommodity(komoditasName);
    setResult(null);
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL_MANUAL}/${komoditasId}`);
      setResult(response.data);
    } catch (error) {
      Alert.alert("Error Server", error.response?.data?.error || "Gagal mengambil prediksi.");
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
    if (!histCommodity || !histYear || !histMonth) {
      Alert.alert("Perhatian", "Pilih komoditas, tahun, dan bulan terlebih dahulu.");
      return;
    }
    setHistLoading(true);
    setHistResult(null);
    try {
      const response = await axios.get(`${API_URL_HISTORY}?komoditas=${histCommodity.id}&tahun=${histYear}&bulan=${histMonth.id}`);
      setHistResult(response.data);
    } catch (error) {
      if (error.response && error.response.status === 404) {
        Alert.alert("Data Kosong", "Tidak ada data pada periode tersebut.");
      } else {
        Alert.alert("Error", "Gagal mengambil data riwayat.");
      }
    } finally {
      setHistLoading(false);
    }
  };

  const formatRupiah = (number) => {
    return new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', minimumFractionDigits: 0 }).format(number);
  };

  const getChartData = () => {
    if (!result || !result.historical_prices) return null;
    const allPrices = [...result.historical_prices, ...result.forecast];
    const nextDatesShort = result.next_dates.map(d => d.split(' ').slice(0, 2).join(' '));
    const allDates = [...result.historical_dates, ...nextDatesShort];

    const labels = allDates.map((date, index) => {
      if (index === 0 || index === 7 || index === 15 || index === 22 || index === 29 || index === 36) return date;
      return "";
    });

    const minVal = Math.min(...allPrices);
    const maxVal = Math.max(...allPrices);
    const padding = (maxVal - minVal) * 0.1 || (minVal * 0.1);
    const paddedMin = minVal - padding;
    const paddedMax = maxVal + padding;

    const dummyData = new Array(allPrices.length).fill(paddedMin);
    dummyData[dummyData.length - 1] = paddedMax;

    return {
      labels: labels,
      datasets: [
        { data: allPrices, color: (opacity = 1) => `rgba(16, 185, 129, ${opacity})`, strokeWidth: 2 },
        { data: dummyData, color: () => 'rgba(0,0,0,0)', strokeWidth: 0, withDots: false }
      ],
    };
  };

  const getHistoryChartData = () => {
    if (!histResult) return null;
    const { dates, prices } = histResult;

    const minVal = Math.min(...prices);
    const maxVal = Math.max(...prices);
    const padding = (maxVal - minVal) * 0.1 || (minVal * 0.1);
    const paddedMin = minVal - padding;
    const paddedMax = maxVal + padding;

    const dummyData = new Array(prices.length).fill(paddedMin);
    dummyData[dummyData.length - 1] = paddedMax;

    const labels = dates.map((date, index) => {
      if (index === 0 || index === 7 || index === 14 || index === 21 || index === 28 || index === dates.length - 1) return date.substring(0, 5);
      return "";
    });

    return {
      labels: labels,
      datasets: [
        { data: prices, color: (opacity = 1) => `rgba(59, 130, 246, ${opacity})`, strokeWidth: 2 },
        { data: dummyData, color: () => 'rgba(0,0,0,0)', strokeWidth: 0, withDots: false }
      ]
    };
  };

  const renderPickerList = () => {
    let data = [];
    let title = "";
    if (pickerType === 'commodity') { data = COMMODITIES; title = "Pilih Komoditas"; }
    else if (pickerType === 'year') { data = YEARS.map(y => ({ id: y, name: y })); title = "Pilih Tahun"; }
    else if (pickerType === 'month') { data = MONTHS; title = "Pilih Bulan"; }
    else if (pickerType === 'home_manual') { data = COMMODITIES; title = "Pilih Komoditas"; }

    return (
      <View style={styles.modalContent}>
        <Text style={styles.modalTitle}>{title}</Text>
        <FlatList
          data={data}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.modalItem}
              onPress={() => {
                if (pickerType === 'commodity') setHistCommodity(item);
                else if (pickerType === 'year') setHistYear(item.id);
                else if (pickerType === 'month') setHistMonth(item);
                else if (pickerType === 'home_manual') predictManual(item.id, item.name);
                setModalVisible(false);
              }}
            >
              <Text style={styles.modalItemText}>{item.name}</Text>
            </TouchableOpacity>
          )}
        />
        <TouchableOpacity style={styles.modalCloseButton} onPress={() => setModalVisible(false)}>
          <Text style={styles.modalCloseText}>Batal</Text>
        </TouchableOpacity>
      </View>
    );
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#f8fafc' }}>
      {/* HEADER */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => setSidebarVisible(true)} style={styles.menuIcon}>
          <Menu color="#1e293b" size={28} />
        </TouchableOpacity>
        <View style={styles.logoContainer}>
          <Leaf color="#10b981" size={28} style={{ marginRight: 6 }} />
          <Text style={styles.headerTitle}>Pangan<Text style={{ color: '#10b981' }}>Predict</Text></Text>
        </View>
        <View style={{ width: 28 }} />
      </View>

      <ScrollView contentContainerStyle={styles.container}>
        {currentScreen === 'home' ? (
          /* HOME SCREEN */
          <>
            <Text style={styles.subtitle}>Sistem Pendeteksi & Peramal Harga Komoditas</Text>
            <View style={styles.imageContainer}>
              {loading ? (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator color="#10b981" size="large" />
                  <Text style={styles.loadingText}>Menganalisis Data...</Text>
                </View>
              ) : imageUri ? (
                <Image source={{ uri: imageUri }} style={styles.imagePreview} />
              ) : selectedCommodity ? (
                <View style={styles.manualSelectedContainer}>
                  <Text style={styles.manualSelectedIcon}>✓</Text>
                  <Text style={styles.manualSelectedText}>{selectedCommodity}</Text>
                  <Text style={styles.manualSelectedSub}>Mode Input Manual Aktif</Text>
                </View>
              ) : (
                <Text style={styles.placeholderText}>Belum ada data / gambar</Text>
              )}
            </View>

            <View style={styles.buttonRow}>
              <TouchableOpacity style={styles.button} onPress={openCamera} disabled={loading}>
                <Text style={styles.buttonText}>Buka Kamera</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.buttonGallery} onPress={openGallery} disabled={loading}>
                <Text style={styles.buttonGalleryText}>Pilih Galeri</Text>
              </TouchableOpacity>
            </View>

            <TouchableOpacity style={styles.buttonManual} onPress={() => openPicker('home_manual')} disabled={loading}>
              <Text style={styles.buttonManualText}>Pilih Komoditas Manual</Text>
            </TouchableOpacity>

            {result && !loading && (
              <View style={styles.resultContainer}>
                <Text style={styles.resultTitle}>{result.display_name}</Text>
                {result.confidence < 1.0 && <Text style={styles.confidence}>Akurasi Pendeteksian AI: {(result.confidence * 100).toFixed(1)}%</Text>}
                {result.confidence === 1.0 && <Text style={styles.confidenceManual}>Sumber Data: Input Manual</Text>}

                <View style={styles.listContainer}>
                  <View style={[styles.listItem, styles.actualItem]}>
                    <Text style={styles.actualDateText}>Hari Ini ({result.last_date})</Text>
                    <Text style={styles.actualPriceText}>{formatRupiah(result.last_price)}</Text>
                  </View>
                  {result.forecast.map((price, index) => (
                    <View key={index} style={styles.listItem}>
                      <Text style={styles.dateText}>{result.next_dates[index]}</Text>
                      <Text style={styles.priceText}>{formatRupiah(price)}</Text>
                    </View>
                  ))}
                </View>

                {result.historical_prices && (
                  <View style={styles.chartContainer}>
                    <Text style={styles.chartTitle}>Tren 30 Hari + Prediksi 7 Hari</Text>
                    <View style={{ position: 'relative', width: '100%' }}>
                      <ScrollView horizontal={true} showsHorizontalScrollIndicator={false} style={{ width: '100%' }}>
                        <LineChart
                          data={getChartData()}
                          width={Dimensions.get("window").width * 1.8}
                          height={220}
                          withInnerLines={false}
                          formatYLabel={(value) => {
                            const val = Math.round(Number(value));
                            if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                            return val.toString();
                          }}
                          chartConfig={{
                            backgroundColor: "#ffffff",
                            backgroundGradientFrom: "#ffffff",
                            backgroundGradientTo: "#ffffff",
                            decimalPlaces: 0,
                            color: (opacity = 1) => `rgba(16, 185, 129, ${opacity})`,
                            labelColor: (opacity = 1) => `rgba(100, 116, 139, ${opacity})`,
                            style: { borderRadius: 16 },
                            propsForDots: { r: "3", strokeWidth: "0" }
                          }}
                          getDotColor={(dataPoint, dataPointIndex) => {
                            if (dataPointIndex > 29) return "#3b82f6"; // Biru untuk prediksi
                            return "#10b981"; // Hijau untuk historis
                          }}
                          bezier
                          style={{ marginTop: 15, borderRadius: 16 }}
                        />
                      </ScrollView>

                      {/* STATIC Y-AXIS OVERLAY */}
                      <View style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 55, backgroundColor: '#ffffff', overflow: 'hidden' }}>
                        <LineChart
                          data={{
                            ...getChartData(),
                            labels: getChartData().labels.map(() => ""), // Hide X labels
                            datasets: getChartData().datasets.map(ds => ({ ...ds, color: () => 'rgba(0,0,0,0)' })) // Hide lines
                          }}
                          width={Dimensions.get("window").width * 1.8}
                          height={220}
                          withInnerLines={false}
                          withDots={false}
                          formatYLabel={(value) => {
                            const val = Math.round(Number(value));
                            if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                            return val.toString();
                          }}
                          chartConfig={{
                            backgroundColor: "#ffffff",
                            backgroundGradientFrom: "#ffffff",
                            backgroundGradientTo: "#ffffff",
                            decimalPlaces: 0,
                            color: (opacity = 1) => `rgba(16, 185, 129, ${opacity})`,
                            labelColor: (opacity = 1) => `rgba(100, 116, 139, ${opacity})`,
                            style: { borderRadius: 16 },
                            propsForDots: { r: "0", strokeWidth: "0" }
                          }}
                          bezier
                          style={{ marginTop: 15, borderRadius: 16 }}
                        />
                      </View>
                    </View>
                    <View style={{ flexDirection: 'row', justifyContent: 'center', marginTop: 10, paddingBottom: 5 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', marginRight: 20 }}>
                        <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: '#10b981', marginRight: 6 }} />
                        <Text style={{ fontSize: 12, color: '#64748b', fontWeight: '500' }}>Harga Historis</Text>
                      </View>
                      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                        <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: '#3b82f6', marginRight: 6 }} />
                        <Text style={{ fontSize: 12, color: '#64748b', fontWeight: '500' }}>Prediksi AI</Text>
                      </View>
                    </View>
                  </View>
                )}
              </View>
            )}
          </>
        ) : (
          /* HISTORY SCREEN */
          <View style={styles.historyContainer}>
            <Text style={styles.historyMainTitle}>Riwayat Harga</Text>
            <Text style={styles.historySubtitle}>Database 9 Tahun (2017-2026)</Text>

            <View style={styles.filterCard}>
              <TouchableOpacity style={styles.filterBtn} onPress={() => openPicker('commodity')}>
                <Text style={styles.filterLabel}>Komoditas</Text>
                <Text style={styles.filterValue}>{histCommodity ? histCommodity.name : "Pilih Komoditas"}</Text>
              </TouchableOpacity>
              <View style={styles.filterRow}>
                <TouchableOpacity style={[styles.filterBtn, { flex: 1, marginRight: 10 }]} onPress={() => openPicker('month')}>
                  <Text style={styles.filterLabel}>Bulan</Text>
                  <Text style={styles.filterValue}>{histMonth ? histMonth.name : "Pilih"}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.filterBtn, { flex: 1 }]} onPress={() => openPicker('year')}>
                  <Text style={styles.filterLabel}>Tahun</Text>
                  <Text style={styles.filterValue}>{histYear ? histYear : "Pilih"}</Text>
                </TouchableOpacity>
              </View>
              <TouchableOpacity style={styles.searchBtn} onPress={fetchHistory} disabled={histLoading}>
                {histLoading ? <ActivityIndicator color="#fff" /> : (
                  <>
                    <Search color="#fff" size={20} style={{ marginRight: 8 }} />
                    <Text style={styles.searchBtnText}>Tampilkan Data</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>

            {histResult && !histLoading && (
              <View style={styles.resultContainer}>
                <Text style={styles.resultTitle}>{histCommodity.name}</Text>
                <Text style={styles.confidenceManual}>{histMonth.name} {histYear}</Text>

                <View style={styles.chartContainer}>
                  <View style={{ position: 'relative', width: '100%' }}>
                    <ScrollView horizontal={true} showsHorizontalScrollIndicator={false} style={{ width: '100%' }}>
                      <LineChart
                        data={getHistoryChartData()}
                        width={Dimensions.get("window").width * 1.8}
                        height={220}
                        withInnerLines={false}
                        formatYLabel={(value) => {
                          const val = Math.round(Number(value));
                          if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                          return val.toString();
                        }}
                        chartConfig={{
                          backgroundColor: "#ffffff",
                          backgroundGradientFrom: "#ffffff",
                          backgroundGradientTo: "#ffffff",
                          decimalPlaces: 0,
                          color: (opacity = 1) => `rgba(59, 130, 246, ${opacity})`,
                          labelColor: (opacity = 1) => `rgba(100, 116, 139, ${opacity})`,
                          style: { borderRadius: 16 },
                          propsForDots: { r: "3", strokeWidth: "0" }
                        }}
                        getDotColor={(dataPoint, dataPointIndex) => {
                          return "#3b82f6"; // Biru solid
                        }}
                        bezier
                        style={{ marginTop: 5, borderRadius: 16 }}
                      />
                    </ScrollView>

                    {/* STATIC Y-AXIS OVERLAY */}
                    <View style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 55, backgroundColor: '#ffffff', overflow: 'hidden' }}>
                      <LineChart
                        data={{
                          ...getHistoryChartData(),
                          labels: getHistoryChartData().labels.map(() => ""), // Hide X labels
                          datasets: getHistoryChartData().datasets.map(ds => ({ ...ds, color: () => 'rgba(0,0,0,0)' })) // Hide lines
                        }}
                        width={Dimensions.get("window").width * 1.8}
                        height={220}
                        withInnerLines={false}
                        withDots={false}
                        formatYLabel={(value) => {
                          const val = Math.round(Number(value));
                          if (val >= 1000) return (val / 1000).toFixed(0) + 'k';
                          return val.toString();
                        }}
                        chartConfig={{
                          backgroundColor: "#ffffff",
                          backgroundGradientFrom: "#ffffff",
                          backgroundGradientTo: "#ffffff",
                          decimalPlaces: 0,
                          color: (opacity = 1) => `rgba(59, 130, 246, ${opacity})`,
                          labelColor: (opacity = 1) => `rgba(100, 116, 139, ${opacity})`,
                          style: { borderRadius: 16 },
                          propsForDots: { r: "0", strokeWidth: "0" }
                        }}
                        bezier
                        style={{ marginTop: 5, borderRadius: 16 }}
                      />
                    </View>
                  </View>
                </View>

                <View style={[styles.listContainer, { marginTop: 20 }]}>
                  <Text style={{ fontWeight: 'bold', marginBottom: 10, color: '#334155' }}>Ringkasan Harga:</Text>
                  <View style={styles.listItem}>
                    <Text style={styles.dateText}>Harga Terendah</Text>
                    <Text style={[styles.priceText, { color: '#10b981' }]}>{formatRupiah(Math.min(...histResult.prices))}</Text>
                  </View>
                  <View style={styles.listItem}>
                    <Text style={styles.dateText}>Harga Tertinggi</Text>
                    <Text style={[styles.priceText, { color: '#ef4444' }]}>{formatRupiah(Math.max(...histResult.prices))}</Text>
                  </View>
                  <View style={styles.listItem}>
                    <Text style={styles.dateText}>Rata-rata</Text>
                    <Text style={styles.priceText}>{formatRupiah(histResult.prices.reduce((a, b) => a + b, 0) / histResult.prices.length)}</Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        )}
      </ScrollView>

      {/* MODAL SIDEBAR */}
      <Modal visible={sidebarVisible} animationType="fade" transparent={true}>
        <View style={styles.sidebarOverlay}>
          <View style={styles.sidebarMenu}>
            <View style={styles.sidebarHeader}>
              <Text style={styles.sidebarTitle}>Menu</Text>
              <TouchableOpacity onPress={() => setSidebarVisible(false)}>
                <X color="#333" size={28} />
              </TouchableOpacity>
            </View>
            <TouchableOpacity style={[styles.sidebarItem, currentScreen === 'home' && styles.sidebarItemActive]}
              onPress={() => { setCurrentScreen('home'); setSidebarVisible(false); }}>
              <Home color={currentScreen === 'home' ? "#10b981" : "#64748b"} size={22} />
              <Text style={[styles.sidebarItemText, currentScreen === 'home' && styles.sidebarItemTextActive]}>Beranda</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.sidebarItem, currentScreen === 'history' && styles.sidebarItemActive]}
              onPress={() => { setCurrentScreen('history'); setSidebarVisible(false); }}>
              <Calendar color={currentScreen === 'history' ? "#10b981" : "#64748b"} size={22} />
              <Text style={[styles.sidebarItemText, currentScreen === 'history' && styles.sidebarItemTextActive]}>Riwayat Harga</Text>
            </TouchableOpacity>
          </View>
          <TouchableOpacity style={styles.sidebarCloseArea} onPress={() => setSidebarVisible(false)} />
        </View>
      </Modal>

      {/* MODAL PICKER */}
      <Modal animationType="slide" transparent={true} visible={modalVisible} onRequestClose={() => setModalVisible(false)}>
        <View style={styles.modalOverlay}>
          {renderPickerList()}
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 50,
    paddingBottom: 15,
    paddingHorizontal: 20,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
  },
  logoContainer: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontSize: 24, fontWeight: '900', color: '#1e293b' },
  menuIcon: { padding: 5 },
  container: { flexGrow: 1, alignItems: 'center', paddingVertical: 30, paddingHorizontal: 20 },
  subtitle: { fontSize: 14, color: '#64748b', marginBottom: 30, textAlign: 'center' },
  imageContainer: { width: 260, height: 260, backgroundColor: '#e2e8f0', borderRadius: 20, justifyContent: 'center', alignItems: 'center', marginBottom: 25, overflow: 'hidden', borderWidth: 3, borderColor: '#cbd5e1', elevation: 4 },
  imagePreview: { width: '100%', height: '100%' },
  placeholderText: { color: '#94a3b8', fontSize: 16, fontWeight: '500' },
  manualSelectedContainer: { alignItems: 'center', padding: 20 },
  manualSelectedIcon: { fontSize: 50, color: '#10b981', marginBottom: 10 },
  manualSelectedText: { fontSize: 22, fontWeight: 'bold', color: '#1e293b', textAlign: 'center' },
  manualSelectedSub: { fontSize: 14, color: '#64748b', marginTop: 5 },
  loadingContainer: { alignItems: 'center' },
  loadingText: { marginTop: 10, fontSize: 16, color: '#10b981', fontWeight: 'bold' },
  buttonRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 15 },
  button: { backgroundColor: '#3b82f6', paddingVertical: 16, borderRadius: 14, flex: 0.48, alignItems: 'center', elevation: 5 },
  buttonGallery: { backgroundColor: '#eab308', paddingVertical: 16, borderRadius: 14, flex: 0.48, alignItems: 'center', elevation: 5 },
  buttonManual: { backgroundColor: '#64748b', paddingVertical: 14, width: '100%', borderRadius: 14, alignItems: 'center', marginBottom: 30 },
  buttonText: { color: '#ffffff', fontSize: 16, fontWeight: 'bold' },
  buttonGalleryText: { color: '#1e293b', fontSize: 16, fontWeight: '900' },
  buttonManualText: { color: '#ffffff', fontSize: 15, fontWeight: 'bold' },
  resultContainer: { width: '100%', backgroundColor: '#ffffff', padding: 24, borderRadius: 20, elevation: 6, alignItems: 'center' },
  resultTitle: { fontSize: 28, fontWeight: '900', color: '#ef4444', marginBottom: 5 },
  confidence: { fontSize: 14, color: '#64748b', marginBottom: 25, fontWeight: '600' },
  confidenceManual: { fontSize: 14, color: '#10b981', marginBottom: 25, fontWeight: 'bold' },
  listContainer: { width: '100%' },
  listItem: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: '#f1f5f9' },
  actualItem: { backgroundColor: '#eff6ff', paddingHorizontal: 16, paddingVertical: 16, borderRadius: 12, borderBottomWidth: 0, marginBottom: 10 },
  dateText: { fontSize: 16, color: '#475569', fontWeight: '500' },
  priceText: { fontSize: 16, color: '#1e293b', fontWeight: '700' },
  actualDateText: { fontSize: 16, color: '#1d4ed8', fontWeight: '800' },
  actualPriceText: { fontSize: 16, color: '#1d4ed8', fontWeight: '900' },
  chartContainer: { marginTop: 25, width: '100%', alignItems: 'center', borderTopWidth: 1, borderTopColor: '#f1f5f9', paddingTop: 20 },
  chartTitle: { fontSize: 16, fontWeight: 'bold', color: '#334155', marginBottom: 10, textAlign: 'center' },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center', padding: 20 },
  modalContent: { width: '100%', backgroundColor: '#fff', borderRadius: 20, padding: 20, maxHeight: '80%' },
  modalTitle: { fontSize: 22, fontWeight: 'bold', color: '#1e293b', marginBottom: 20, textAlign: 'center' },
  modalItem: { paddingVertical: 15, borderBottomWidth: 1, borderBottomColor: '#f1f5f9' },
  modalItemText: { fontSize: 18, color: '#334155', textAlign: 'center' },
  modalCloseButton: { marginTop: 20, backgroundColor: '#ef4444', padding: 15, borderRadius: 12, alignItems: 'center' },
  modalCloseText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },

  /* Sidebar Styles */
  sidebarOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', flexDirection: 'row' },
  sidebarMenu: { width: 280, backgroundColor: '#fff', height: '100%', padding: 20, paddingTop: 50 },
  sidebarCloseArea: { flex: 1 },
  sidebarHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 40 },
  sidebarTitle: { fontSize: 24, fontWeight: 'bold', color: '#1e293b' },
  sidebarItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 15, paddingHorizontal: 15, borderRadius: 12, marginBottom: 10 },
  sidebarItemActive: { backgroundColor: '#ecfdf5' },
  sidebarItemText: { fontSize: 18, color: '#64748b', marginLeft: 15, fontWeight: '600' },
  sidebarItemTextActive: { color: '#10b981', fontWeight: 'bold' },

  /* History Styles */
  historyContainer: { width: '100%' },
  historyMainTitle: { fontSize: 24, fontWeight: '900', color: '#1e293b', textAlign: 'center' },
  historySubtitle: { fontSize: 14, color: '#64748b', textAlign: 'center', marginBottom: 30 },
  filterCard: { backgroundColor: '#fff', borderRadius: 20, padding: 20, elevation: 4, width: '100%', marginBottom: 30 },
  filterRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 15 },
  filterBtn: { backgroundColor: '#f1f5f9', padding: 15, borderRadius: 12, marginBottom: 15 },
  filterLabel: { fontSize: 12, color: '#64748b', fontWeight: 'bold', marginBottom: 5 },
  filterValue: { fontSize: 16, color: '#1e293b', fontWeight: 'bold' },
  searchBtn: { backgroundColor: '#3b82f6', padding: 16, borderRadius: 12, flexDirection: 'row', justifyContent: 'center', alignItems: 'center' },
  searchBtnText: { color: '#fff', fontSize: 16, fontWeight: 'bold' }
});
