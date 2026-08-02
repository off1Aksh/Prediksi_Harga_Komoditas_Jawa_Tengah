import matplotlib.pyplot as plt
import numpy as np

# Data kelas komoditas
classes = [
    'Bawang Merah', 'Bawang Putih', 'Beras', 'Cabai Merah', 
    'Cabai Rawit', 'Daging Ayam', 'Minyak Goreng', 'Telur Ayam'
]

# Jumlah gambar asli (Sebelum) dan target (Sesudah)
sebelum = [500, 500, 4237, 216, 172, 500, 160, 500]
sesudah = [100, 100, 100, 100, 100, 100, 100, 100]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, sebelum, width, label='Sebelum Undersampling', color='#e74c3c')
rects2 = ax.bar(x + width/2, sesudah, width, label='Sesudah Undersampling', color='#2ecc71')

ax.set_ylabel('Jumlah Gambar', fontsize=12)
ax.set_title('Perbandingan Distribusi Gambar per Kelas Sebelum dan Sesudah Undersampling', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Menambahkan angka di atas batang grafik
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
out_path = r'd:\KULIAH\SKRIPSI\grafik_distribusi_citra.png'
plt.savefig(out_path, dpi=300)
print(f"\nGrafik distribusi berhasil disimpan di:\n{out_path}")
