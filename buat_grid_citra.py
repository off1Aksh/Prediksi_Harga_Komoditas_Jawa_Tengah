import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_dir = r"d:\KULIAH\SKRIPSI\Dataset\gambar_balanced"

# Spesifikasi nama file gambar yang dipilih oleh user
selected_images = {
    "bawang_merah": "dup_1bcfe8f6f00740aca19a287f704f0956.jpg",
    "bawang_putih": "dup_684c047aa96f463dad0e1cf0100d2ea8.jpg",
    "beras": "pngtree-rice-grain-flower-arrangement-isolated-png-image_19386831.webp",
    "cabai_merah": "dup_1c2ae78e174f4268af8bcf1e6db2e09a - Copy - Copy - Copy - Copy.jpg",
    "cabai_rawit": "dup_28ebc41c6fe9487f95d3a320b9369937 - Copy - Copy.jpg",
    "daging_ayam": "00000379.jpg",
    "minyak_goreng": "images98698.jpg",
    "telur_ayam": "00000063.jpg" # Folder bernama telur_ayam
}

# Daftar kelas berdasarkan urutan keys di dictionary
classes = list(selected_images.keys())

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i, cls in enumerate(classes):
    img_filename = selected_images[cls]
    img_path = os.path.join(dataset_dir, cls, img_filename)
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Konversi ke RGB untuk menghindari error jika gambar format CMYK/RGBA
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize ke 224x224 (Input Shape MobileNetV2)
        img = img.resize((224, 224))
            
        axes[i].imshow(img)
    else:
        print(f"Peringatan: Gambar tidak ditemukan -> {img_path}")
        
    # Format judul agar rapi (contoh: bawang_merah -> Bawang Merah)
    title = cls.replace('_', ' ').title()
    axes[i].set_title(title, fontsize=14, fontweight='bold')
    axes[i].axis('off') # Hilangkan garis axis X dan Y

plt.suptitle("Sampel Citra Setelah Penyesuaian Ukuran (Resize ke 224x224)", fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

out_path = r"d:\KULIAH\SKRIPSI\sampel_dataset_citra_resized.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nGambar sampel dataset berhasil disimpan di:\n{out_path}")
