import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# 1. Buat contoh matriks piksel acak ukuran 3x3 (rentang 0 - 255)
np.random.seed(42)
raw_pixels = np.array([
    [255, 128, 0],
    [50,  200, 100],
    [10,  250, 127]
])

# 2. Hitung nilai normalisasi MobileNetV2: (x / 127.5) - 1
norm_pixels = (raw_pixels / 127.5) - 1.0
norm_pixels = np.round(norm_pixels, 2) # Bulatkan 2 angka di belakang koma untuk tampilan

# 3. Setup Kanvas Gambar
fig, (ax1, ax_arrow, ax2) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 0.4, 1]})

# Fungsi bantuan untuk menggambar tabel/matriks piksel
def draw_matrix(ax, data, title, cmap):
    # Tampilkan sebagai heatmap
    cax = ax.matshow(data, cmap=cmap, vmin=np.min(data), vmax=np.max(data))
    # Sembunyikan axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
    
    # Tambahkan angka teks di dalam setiap kotak piksel
    for (i, j), z in np.ndenumerate(data):
        # Tentukan warna teks (hitam/putih) agar kontras dengan warna background heatmap
        text_color = 'white' if (z < np.mean(data)) else 'black'
        ax.text(j, i, f'{z}', ha='center', va='center', fontsize=14, color=text_color, fontweight='bold')

# Kotak Kiri: Nilai RGB Asli
draw_matrix(ax1, raw_pixels, "Nilai Piksel Asli (Original RGB)\nRentang: [0, 255]", 'Blues')

# Tengah: Panah Proses
ax_arrow.axis('off')
arrow = FancyArrowPatch((0.1, 0.5), (0.9, 0.5), mutation_scale=20, color='gray')
ax_arrow.add_patch(arrow)
ax_arrow.text(0.5, 0.6, 'Fungsi:\npreprocess_input()', ha='center', va='bottom', fontsize=11, fontweight='bold', color='gray')
ax_arrow.text(0.5, 0.4, 'x = (x / 127.5) - 1', ha='center', va='top', fontsize=10, style='italic', color='gray')

# Kotak Kanan: Nilai Ternormalisasi
draw_matrix(ax2, norm_pixels, "Piksel Ternormalisasi (MobileNetV2)\nRentang: [-1, 1]", 'Greens')

plt.suptitle("Ilustrasi Transformasi Nilai Matriks Piksel pada Pra-pemrosesan Citra", fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()

# Simpan Gambar
out_path = r"d:\KULIAH\SKRIPSI\ilustrasi_normalisasi_piksel.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Berhasil! Gambar ilustrasi normalisasi piksel telah disimpan di:\n{out_path}")
