
import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4fikOeote8mjCKla_BMf8kVsDweNygzThQWtIctZdgxMF2ecvm6r58nWe&s=10"

request = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(request) as response:
    data = response.read()

image = cv2.imdecode(
    np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
)

if image is None:
    print("Image not found")
else:
    blur_image = cv2.GaussianBlur(image, (7, 7), 0)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

