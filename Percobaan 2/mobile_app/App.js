import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ActivityIndicator, ScrollView, Alert, Modal, FlatList } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Leaf } from 'lucide-react-native';

// GANTI INI DENGAN IP ADDRESS LAPTOP ANDA (Cek via ipconfig di CMD)
const API_URL_IMAGE = "http://172.16.161.127:5000/api/predict-image";
const API_URL_MANUAL = "http://172.16.161.127:5000/api/predict-single";

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

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // Modal state
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedCommodity, setSelectedCommodity] = useState(null);

  const openCamera = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (permissionResult.granted === false) {
      Alert.alert("Izin Ditolak", "Anda harus mengizinkan akses kamera.");
      return;
    }
    const pickerResult = await ImagePicker.launchCameraAsync({
      allowsEditing: true, aspect: [1, 1], quality: 0.8,
    });

    if (!pickerResult.canceled) {
      const uri = pickerResult.assets[0].uri;
      setImageUri(uri);
      setSelectedCommodity(null);
      setResult(null);
      // Langsung proses gambar
      analyzeImage(uri);
    }
  };

  const openGallery = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permissionResult.granted === false) {
      Alert.alert("Izin Ditolak", "Anda harus mengizinkan akses galeri.");
      return;
    }
    const pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true, aspect: [1, 1], quality: 0.8,
    });

    if (!pickerResult.canceled) {
      const uri = pickerResult.assets[0].uri;
      setImageUri(uri);
      setSelectedCommodity(null);
      setResult(null);
      // Langsung proses gambar
      analyzeImage(uri);
    }
  };

  // Prediksi via Gambar AI
  const analyzeImage = async (uri) => {
    if (API_URL_IMAGE.includes("x.x")) {
      Alert.alert("IP Address Salah!", "Ganti tulisan '192.168.x.x' di App.js dengan IP laptop Anda.");
      return;
    }

    setLoading(true);
    const filename = uri.split('/').pop();
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image`;

    const formData = new FormData();
    formData.append('image', { uri: uri, name: filename, type });

    try {
      const response = await axios.post(API_URL_IMAGE, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      if (error.response) {
        Alert.alert("Error Server", error.response.data.error || "Gagal memproses gambar.");
      } else {
        Alert.alert("Error Koneksi", "Tidak bisa terhubung ke server.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Prediksi Manual (Langsung Dipanggil dari Modal)
  const predictManual = async (komoditasId, komoditasName) => {
    setModalVisible(false);
    setImageUri(null);
    setSelectedCommodity(komoditasName);
    setResult(null);

    if (API_URL_MANUAL.includes("x.x")) {
      Alert.alert("IP Address Salah!", "Ganti tulisan '192.168.x.x' di App.js dengan IP laptop Anda.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.get(`${API_URL_MANUAL}/${komoditasId}`);
      setResult(response.data);
    } catch (error) {
      if (error.response) {
        Alert.alert("Error Server", error.response.data.error || "Gagal mengambil prediksi.");
      } else {
        Alert.alert("Error Koneksi", "Tidak bisa terhubung ke server.");
      }
    } finally {
      setLoading(false);
    }
  };

  const formatRupiah = (number) => {
    return new Intl.NumberFormat('id-ID', {
      style: 'currency',
      currency: 'IDR',
      minimumFractionDigits: 0
    }).format(number);
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.logoContainer}>
        <Leaf color="#10b981" size={32} style={{ marginRight: 8 }} />
        <Text style={styles.headerTitle}>
          Pangan<Text style={{ color: '#10b981' }}>Predict</Text>
        </Text>
      </View>
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

      <TouchableOpacity style={styles.buttonManual} onPress={() => setModalVisible(true)} disabled={loading}>
        <Text style={styles.buttonManualText}>Pilih Komoditas Manual</Text>
      </TouchableOpacity>

      {result && !loading && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>{result.display_name}</Text>

          {result.confidence < 1.0 && (
            <Text style={styles.confidence}>Akurasi Pendeteksian AI: {(result.confidence * 100).toFixed(1)}%</Text>
          )}
          {result.confidence === 1.0 && (
            <Text style={styles.confidenceManual}>Sumber Data: Input Manual</Text>
          )}

          <View style={styles.listContainer}>
            <View style={[styles.listItem, styles.actualItem]}>
              <Text style={styles.actualDateText}>Aktual ({result.last_date})</Text>
              <Text style={styles.actualPriceText}>{formatRupiah(result.last_price)}</Text>
            </View>

            {result.forecast.map((price, index) => (
              <View key={index} style={styles.listItem}>
                <Text style={styles.dateText}>{result.next_dates[index]}</Text>
                <Text style={styles.priceText}>{formatRupiah(price)}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Modal Pemilihan Manual */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Pilih Komoditas</Text>
            <FlatList
              data={COMMODITIES}
              keyExtractor={(item) => item.id}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={styles.modalItem}
                  onPress={() => predictManual(item.id, item.name)}
                >
                  <Text style={styles.modalItemText}>{item.name}</Text>
                </TouchableOpacity>
              )}
            />
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setModalVisible(false)}
            >
              <Text style={styles.modalCloseText}>Batal</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#f8fafc',
    alignItems: 'center',
    paddingVertical: 60,
    paddingHorizontal: 20,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 5,
  },
  headerTitle: {
    fontSize: 30,
    fontWeight: '900',
    color: '#1e293b',
  },
  subtitle: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 30,
    textAlign: 'center'
  },
  imageContainer: {
    width: 260,
    height: 260,
    backgroundColor: '#e2e8f0',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 25,
    overflow: 'hidden',
    borderWidth: 3,
    borderColor: '#cbd5e1',
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  imagePreview: {
    width: '100%',
    height: '100%',
  },
  placeholderText: {
    color: '#94a3b8',
    fontSize: 16,
    fontWeight: '500'
  },
  manualSelectedContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  manualSelectedIcon: {
    fontSize: 50,
    color: '#10b981',
    marginBottom: 10,
  },
  manualSelectedText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1e293b',
    textAlign: 'center',
  },
  manualSelectedSub: {
    fontSize: 14,
    color: '#64748b',
    marginTop: 5,
  },
  loadingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#10b981',
    fontWeight: 'bold',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 15,
  },
  button: {
    backgroundColor: '#3b82f6',
    paddingVertical: 16,
    paddingHorizontal: 15,
    borderRadius: 14,
    flex: 0.48,
    alignItems: 'center',
    shadowColor: "#3b82f6",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  buttonGallery: {
    backgroundColor: '#eab308',
    paddingVertical: 16,
    paddingHorizontal: 15,
    borderRadius: 14,
    flex: 0.48,
    alignItems: 'center',
    shadowColor: "#eab308",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  buttonManual: {
    backgroundColor: '#64748b',
    paddingVertical: 14,
    width: '100%',
    borderRadius: 14,
    alignItems: 'center',
    marginBottom: 30,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  buttonGalleryText: {
    color: '#1e293b',
    fontSize: 16,
    fontWeight: '900',
  },
  buttonManualText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: 'bold',
  },
  resultContainer: {
    width: '100%',
    backgroundColor: '#ffffff',
    padding: 24,
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 6,
    alignItems: 'center',
  },
  resultTitle: {
    fontSize: 28,
    fontWeight: '900',
    color: '#ef4444',
    marginBottom: 5,
  },
  confidence: {
    fontSize: 14,
    color: '#64748b',
    marginBottom: 25,
    fontWeight: '600'
  },
  confidenceManual: {
    fontSize: 14,
    color: '#10b981',
    marginBottom: 25,
    fontWeight: 'bold'
  },
  listContainer: {
    width: '100%',
  },
  listItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  actualItem: {
    backgroundColor: '#eff6ff',
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderRadius: 12,
    borderBottomWidth: 0,
    marginBottom: 10,
  },
  dateText: {
    fontSize: 16,
    color: '#475569',
    fontWeight: '500',
  },
  priceText: {
    fontSize: 16,
    color: '#1e293b',
    fontWeight: '700',
  },
  actualDateText: {
    fontSize: 16,
    color: '#1d4ed8',
    fontWeight: '800',
  },
  actualPriceText: {
    fontSize: 16,
    color: '#1d4ed8',
    fontWeight: '900',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 20,
    maxHeight: '80%',
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 20,
    textAlign: 'center',
  },
  modalItem: {
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  modalItemText: {
    fontSize: 18,
    color: '#334155',
    textAlign: 'center',
  },
  modalCloseButton: {
    marginTop: 20,
    backgroundColor: '#ef4444',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  modalCloseText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  }
});
