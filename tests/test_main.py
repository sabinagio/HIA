import requests

def test_chat_endpoint():
    url = "http://127.0.0.1:8000/chat"
    input_query = {
        "message": "Where can I get food assistance for my family of 4?",
        "location": "Amsterdam"  # Optional but helpful
    }

    try:
        response = requests.post(url, json=input_query)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")


if __name__ == "__main__":
    test_chat_endpoint()
