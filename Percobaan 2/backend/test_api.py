import requests

# Masukkan path gambar cabai atau telur yang ada di laptop Anda
# Contoh: r"d:\KULIAH\SKRIPSI\Dataset\gambar\Cabai_Rawit\Image_1.jpg"
IMAGE_PATH = r"d:\KULIAH\SKRIPSI\Dataset\gambar\Cabai_Rawit\test cabe.jpg"

url = "http://localhost:5000/api/predict-image"

print(f"Mengirim gambar {IMAGE_PATH} ke API...")

try:
    with open(IMAGE_PATH, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    import json
    print(json.dumps(response.json(), indent=4))
    
except Exception as e:
    print(f"Error: {e}")
