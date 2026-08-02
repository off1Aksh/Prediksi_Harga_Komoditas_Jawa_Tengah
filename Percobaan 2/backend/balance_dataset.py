"""
Balance Dataset — Undersample ke 100 Gambar per Kelas
=====================================================
Jalankan script ini di LOKAL sebelum upload ke Kaggle.

Cara pakai:
1. Jalankan: python balance_dataset.py
2. Folder output: d:\KULIAH\SKRIPSI\Dataset\gambar_balanced\
3. Zip folder gambar_balanced lalu upload ke Kaggle Dataset
4. Dataset asli TIDAK diubah

Output:
  gambar_balanced/
  ├── bawang_merah/   (100 gambar)
  ├── bawang_putih/   (100 gambar)
  ├── beras/          (100 gambar)
  ├── cabai_merah/    (100 gambar)
  ├── cabai_rawit/    (100 gambar)
  ├── daging_ayam/    (100 gambar)
  ├── minyak_goreng/  (100 gambar)
  └── telur_ayam/     (100 gambar)
"""

import os
import shutil
import random

# ============================================================
# KONFIGURASI
# ============================================================
SEED = 42
TARGET_PER_CLASS = 100
SOURCE_DIR = r"d:\KULIAH\SKRIPSI\Dataset\gambar"
OUTPUT_DIR = r"d:\KULIAH\SKRIPSI\Dataset\gambar_balanced"

# Ekstensi gambar yang valid
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ============================================================
# MAIN
# ============================================================
def main():
    random.seed(SEED)

    # Hapus output lama jika ada
    if os.path.exists(OUTPUT_DIR):
        print(f"Menghapus folder output lama: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dapatkan daftar kelas (folder)
    class_names = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    print(f"Sumber dataset : {SOURCE_DIR}")
    print(f"Output dataset : {OUTPUT_DIR}")
    print(f"Target per kelas: {TARGET_PER_CLASS}")
    print(f"Kelas ditemukan : {len(class_names)}")
    print(f"Seed           : {SEED}")
    print(f"{'='*50}")

    total_copied = 0

    for class_name in class_names:
        src_class_dir = os.path.join(SOURCE_DIR, class_name)
        dst_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        # Filter hanya file gambar
        all_images = [
            f for f in os.listdir(src_class_dir)
            if os.path.isfile(os.path.join(src_class_dir, f))
            and os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]

        original_count = len(all_images)

        if original_count < TARGET_PER_CLASS:
            print(f"  ⚠ {class_name}: hanya {original_count} gambar "
                  f"(kurang dari target {TARGET_PER_CLASS}). Semua gambar di-copy.")
            selected = all_images
        else:
            selected = random.sample(all_images, TARGET_PER_CLASS)

        # Copy gambar terpilih
        for img_name in selected:
            src_path = os.path.join(src_class_dir, img_name)
            dst_path = os.path.join(dst_class_dir, img_name)
            shutil.copy2(src_path, dst_path)

        print(f"  ✅ {class_name}: {original_count} → {len(selected)} gambar")
        total_copied += len(selected)

    print(f"{'='*50}")
    print(f"Total gambar: {total_copied} ({total_copied // len(class_names)} per kelas)")
    print(f"Dataset balanced tersimpan di: {OUTPUT_DIR}")
    print(f"\nLangkah selanjutnya:")
    print(f"  1. Zip folder '{os.path.basename(OUTPUT_DIR)}'")
    print(f"  2. Upload ke Kaggle sebagai Dataset baru")
    print(f"  3. Gunakan di notebook Kaggle dengan path:")
    print(f"     /kaggle/input/<nama-dataset-anda>/gambar_balanced/")


if __name__ == "__main__":
    main()
