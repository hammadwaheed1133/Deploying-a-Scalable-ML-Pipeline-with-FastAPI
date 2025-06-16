# local_api.py

import requests

# Base URL of the local FastAPI server
url = "http://127.0.0.1:8000"

# TODO: send a GET using the URL
response_get = requests.get(url)
print("Status Code:", response_get.status_code)
print("Result:", response_get.json()["message"])

# TODO: send a POST and print the status code and result
data = {
    "age": 45,
    "education": "Bachelors",
    "occupation": "Engineer"
}

response_post = requests.post(f"{url}/predict", json=data)
print("Status Code:", response_post.status_code)
print("Result:", response_post.json()["result"])
