
# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Read input image from URL
url = "https://www.yamaha-motor-india.com/theme/v4/images/webp_images/r_series_all/r2/color/r2_racing_blue.webp"

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

    # Apply morphological erosion
    eroded = cv2.erode(
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

    # Display eroded image
    plt.subplot(1, 2, 2)
    plt.imshow(eroded, cmap="gray")
    plt.title("Eroded Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Morphological erosion completed successfully.")
