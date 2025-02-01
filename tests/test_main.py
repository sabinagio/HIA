import requests
import json


def test_chat():
    url = "http://127.0.0.1:8000/chat"
    input_query = {
        "message": "Where can I get food assistance for my family of 4?",
        "location": "Amsterdam"
    }

    print("\n--- Sending Request ---")
    print(f"URL: {url}")
    print(f"Input: {json.dumps(input_query, indent=2)}")

    response = requests.post(url, json=input_query)

    print("\n--- Got Response ---")
    print(f"Status: {response.status_code}")
    print(f"Full Response: {response.text}")

    if response.status_code != 200:
        print(f"\nError Details: {response.text}")


if __name__ == "__main__":
    test_chat()