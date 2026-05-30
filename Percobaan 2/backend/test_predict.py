import requests
import numpy as np

# Generate dummy historical data (14 days, 8 commodities)
dummy_data = np.random.rand(14, 8) * 1000 + 10000 # Prices around 10k-11k

payload = {
    "historical_data": dummy_data.tolist()
}

response = requests.post("http://localhost:5000/api/predict", json=payload)

if response.status_code == 200:
    print("Success!")
    data = response.json()
    forecast = np.array(data['forecast'])
    print(f"Forecast shape: {forecast.shape}")
else:
    print(f"Error {response.status_code}: {response.text}")
