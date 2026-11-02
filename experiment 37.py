
import cv2
import numpy as np
import urllib.request
import os

# Image URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4fikOeote8mjCKla_BMf8kVsDweNygzThQWtIctZdgxMF2ecvm6r58nWe&s=10"

# Download image
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
    print("Image could not be loaded.")
    raise SystemExit

# Convert BGR image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define foreground color range (green)
lower_color = np.array([35, 50, 50])
upper_color = np.array([85, 255, 255])

# Create foreground mask
mask = cv2.inRange(hsv, lower_color, upper_color)

# Extract foreground
foreground = cv2.bitwise_and(image, image, mask=mask)

# Get program folder
folder = os.path.dirname(os.path.abspath(__file__))

# Save output
cv2.imwrite(os.path.join(folder, "foreground_image.jpg"), foreground)
cv2.imwrite(os.path.join(folder, "foreground_mask.jpg"), mask)

# Display output
cv2.imshow("Original Image", image)
cv2.imshow("Foreground Extraction", foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Open output folder
os.startfile(folder)
