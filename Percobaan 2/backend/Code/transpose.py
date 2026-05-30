import pandas as pd
import os

# konfigurasi: (file_excel, [(nama_komoditas, nama_kolom_csv), ...])
configs = [
    ("bawang_merah_putih_2021_2026.xlsx", [
        ("Bawang Merah", "bawang_merah"),
        ("Bawang Putih", "bawang_putih"),
    ]),
    ("beras 2021-2026.xlsx", [
        ("Beras", "beras"),
    ]),
    ("cabai merah dan cabe rawit 2021-2026.xlsx", [
        ("Cabai Merah", "cabai_merah"),
        ("Cabai Rawit", "cabai_rawit"),
    ]),
    ("daging ayam 2021-2026.xlsx", [
        ("Daging Ayam", "daging_ayam"),
    ]),
    ("minyak goreng 2021-2026.xlsx", [
        ("Minyak Goreng", "minyak_goreng"),
    ]),
    ("telur ayam 2021-2026.xlsx", [
        ("Telur Ayam", "telur_ayam"),
    ]),
]

# direktori script ini berada
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "Dataset Final")
os.makedirs(output_dir, exist_ok=True)

# proses semua file dan kumpulkan hasilnya
all_dfs = []

for file_name, commodities in configs:
    file_path = os.path.join(script_dir, file_name)
    df = pd.read_excel(file_path)

    for komoditas, col_name in commodities:
        # ambil baris parent category
        row = df[df["Komoditas (Rp)"] == komoditas]

        # hapus kolom No dan Komoditas (Rp)
        row = row.drop(columns=["No", "Komoditas (Rp)"])

        # transpose: tanggal jadi baris
        transposed = row.T
        transposed.columns = [col_name]
        transposed = transposed.reset_index().rename(columns={"index": "tanggal"})

        all_dfs.append(transposed)
        print(f"[OK] {komoditas} dari {file_name} ({len(transposed)} baris)")

# gabungkan semua dataframe — semua memiliki urutan tanggal yang sama
# ambil kolom tanggal dari df pertama, lalu concat kolom komoditas saja
tanggal = all_dfs[0][["tanggal"]]
commodity_cols = [df.drop(columns=["tanggal"]) for df in all_dfs]
result = pd.concat([tanggal] + commodity_cols, axis=1)

# hapus baris yang memiliki missing value ("-")
before = len(result)
result = result[~(result == "-").any(axis=1)].reset_index(drop=True)
after = len(result)
print(f"\nMenghapus {before - after} baris dengan missing value")
print(f"Hasil gabungan: {after} baris x {result.shape[1]} kolom")
print(f"Kolom: {result.columns.tolist()}")

# simpan ke satu file csv
output_path = os.path.join(output_dir, "dataset_harga_komoditas_2021_2026.csv")
result.to_csv(output_path, index=False)
print(f"\n[OK] Dataset berhasil disimpan ke: {output_path}")
print(result.head())