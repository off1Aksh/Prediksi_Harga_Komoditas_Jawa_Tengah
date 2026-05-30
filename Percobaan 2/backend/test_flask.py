import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app
import json

client = app.test_client()

print("========================================")
print("1. Testing Endpoint GET /api/latest-data")
print("========================================")
response = client.get('/api/latest-data')
print(f"Status Code: {response.status_code}")
data = json.loads(response.data)
if response.status_code == 200:
    print(f"Keys in Response: {list(data.keys())}")
    print(f"Historical Data Rows: {len(data.get('historical_data', []))}")
    if data.get('historical_data'):
        print(f"Columns per row: {len(data['historical_data'][0])}")
else:
    print(f"Error Response: {data}")

print("\n========================================")
print("2. Testing Endpoint POST /api/predict")
print("========================================")
response = client.post('/api/predict')
print(f"Status Code: {response.status_code}")
data = json.loads(response.data)
if response.status_code == 200:
    print(f"Message: {data.get('message')}")
    forecast = data.get('forecast', [])
    print(f"Forecast Rows (Days): {len(forecast)}")
    if forecast:
        print(f"Columns per row (Commodities): {len(forecast[0])}")
        print(f"Sample Forecast Day 1 (Rupiah): {forecast[0]}")
else:
    print(f"Error Response: {data}")
