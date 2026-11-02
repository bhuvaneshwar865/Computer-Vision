import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os

# Get the current folder
folder = os.path.dirname(os.path.abspath(__file__))

# Read image from URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4fikOeote8mjCKla_BMf8kVsDweNygzThQWtIctZdgxMF2ecvm6r58nWe&s=10"

request = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(request, timeout=30) as response:
    data = response.read()

image = cv2.imdecode(
    np.frombuffer(data, dtype=np.uint8),
    cv2.IMREAD_COLOR
)

if image is None:
    print("Error: Image could not be loaded.")
else:
    # Convert image into grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Apply binary thresholding
    threshold_value = 127

    ret, segmented = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Apply inverse thresholding
    ret, inverse = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Save segmented images
    cv2.imwrite(
        os.path.join(folder, "segmented_image.jpg"),
        segmented
    )

    cv2.imwrite(
        os.path.join(folder, "inverse_segmented_image.jpg"),
        inverse
    )

    # Display results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(segmented, cmap="gray")
    plt.title("Binary Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(inverse, cmap="gray")
    plt.title("Inverse Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Image segmentation completed successfully.")

# Open File Explorer
os.startfile(folder)
