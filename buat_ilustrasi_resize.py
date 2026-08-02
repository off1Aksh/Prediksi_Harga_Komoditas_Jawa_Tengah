import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# Path gambar asli yang akan dijadikan contoh (menggunakan gambar cabai merah yang sebelumnya Anda sebutkan)
img_path = r"d:\KULIAH\SKRIPSI\Dataset\gambar\cabai_merah\dup_1c2ae78e174f4268af8bcf1e6db2e09a - Copy - Copy - Copy - Copy.jpg"

# Fallback (cadangan) jika file di atas tidak ditemukan di folder tersebut
if not os.path.exists(img_path):
    dataset_dir = r"d:\KULIAH\SKRIPSI\Dataset\gambar_balanced"
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                img_path = os.path.join(root, f)
                break
        if os.path.exists(img_path): 
            break

try:
    img_ori = Image.open(img_path)
    if img_ori.mode != 'RGB':
        img_ori = img_ori.convert('RGB')
        
    width_ori, height_ori = img_ori.size
    
    # Lakukan proses resize ke ukuran 224x224
    img_resized = img_ori.resize((224, 224))
    
    # Buat figure kanvas
    fig = plt.figure(figsize=(11, 5))
    
    # Kotak 1: Gambar Asli (Posisi Kiri)
    ax1 = fig.add_axes([0.05, 0.1, 0.35, 0.75])
    ax1.imshow(img_ori)
    ax1.set_title(f"Gambar Citra Asli (Kamera/Internet)\nDimensi: {width_ori} × {height_ori} piksel", fontsize=12, pad=10)
    ax1.axis('off')
    
    # Kotak 2: Ruang untuk tanda panah di tengah
    ax_arrow = fig.add_axes([0.4, 0.1, 0.2, 0.75])
    ax_arrow.axis('off')
    arrow = FancyArrowPatch((0.1, 0.5), (0.9, 0.5), mutation_scale=30, color='#34495e')
    ax_arrow.add_patch(arrow)
    ax_arrow.text(0.5, 0.55, 'Proses Resizing', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#34495e')
    
    # Kotak 3: Gambar Hasil Resize (Posisi Kanan)
    # Diberi outline kotak samar agar ukuran persegi 224x224 nya terlihat tegas
    ax2 = fig.add_axes([0.6, 0.1, 0.35, 0.75])
    ax2.imshow(img_resized)
    ax2.set_title(f"Target Masukan Model MobileNetV2\nDimensi: 224 × 224 × 3", fontsize=12, pad=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # Judul utama gambar
    plt.suptitle("Ilustrasi Proses Penyesuaian Ukuran Citra (Image Resizing)", fontsize=16, fontweight='bold', y=0.98)
    
    # Simpan sebagai file PNG
    out_path = r"d:\KULIAH\SKRIPSI\ilustrasi_resizing_citra.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Berhasil! Gambar ilustrasi telah disimpan di:\n{out_path}")
    
except Exception as e:
    print(f"Terjadi kesalahan saat memproses gambar: {e}")
