
# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Read input image from URL
url = "https://sm.ign.com/ign_za/cover/m/marvels-sp/marvels-spider-man-remastered_az82.jpg"

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

    # Apply dilation operation
    dilated = cv2.dilate(
        gray,
        kernel,
        iterations=1
    )

    # Display original grayscale image
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Grayscale Image")
    plt.axis("off")

    # Display dilated image
    plt.subplot(1, 2, 2)
    plt.imshow(dilated, cmap="gray")
    plt.title("Dilated Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Morphological dilation completed successfully.")
