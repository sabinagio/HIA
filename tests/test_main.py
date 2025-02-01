import requests

url = "http://127.0.0.1:8000/chat"
input_query = {
    "message": "Where can I get food assistance for my family of 4?"
}

response = requests.post(url, json=input_query)
print(response.json())
