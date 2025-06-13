import requests

response = requests.post('http://127.0.0.1:5000/predict', 
                       json={'city': 'kampala', 'country': 'uganda', 'date': '2025-06-14'})
print(response.json())