import requests

url = "http://3.70.232.239:80/v1/models"

response = requests.get(url, timeout=60)

if response.status_code == 200:
    # Request was successful, print the response content
    print(response.text)
else:
    # Request failed with an error status code, print the status code
    print(f"Request failed with status code: {response.status_code}")
