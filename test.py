import requests

URL = "https://server-chocho-production.up.railway.app/ocr"
IMAGE_PATH = "test1.jpg"

with open(IMAGE_PATH, "rb") as f:
    files = {
        "file": ("test1.jpg", f, "image/png")
    }

    response = requests.post(URL, files=files)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())

import requests

URL = "https://server-chocho-production.up.railway.app/scan"
IMAGE_PATH = "test2.png"

with open(IMAGE_PATH, "rb") as f:
    files = {
        "file": ("test2.png", f, "image/png")
    }

    response = requests.post(URL, files=files)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
