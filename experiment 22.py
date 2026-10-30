
# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Read input image from URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4fikOeote8mjCKla_BMf8kVsDweNygzThQWtIctZdgxMF2ecvm6r58nWe&s=10"

request = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(request, timeout=30) as response:
    data = response.read()

image = cv2.imdecode(
    np.frombuffer(data, dtype=np.uint8),
    cv2.IMREAD_COLOR
)

if image is None:
    print("Error: Image not found.")
else:
    # Convert image into grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Create structuring element
    kernel = np.ones((5, 5), np.uint8)

    # Apply Top Hat morphological operation
    tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        kernel
    )

    # Display images
    plt.figure(figsize=(10, 4))

    # Original grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale Image")
    plt.axis("off")

    # Top Hat image
    plt.subplot(1, 2, 2)
    plt.imshow(tophat, cmap="gray")
    plt.title("Top Hat Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Top Hat operation completed successfully.")
