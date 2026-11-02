import cv2
import numpy as np
import urllib.request
import os

# Get current script folder
folder = os.path.dirname(os.path.abspath(__file__))

# Download input image
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
    print("Image not found!")
    raise SystemExit

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade Classifier
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Face cascade file could not be loaded!")
    raise SystemExit

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

# Count faces
count = len(faces)
print("Number of Faces Detected:", count)

# Draw rectangle around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Display face count on image
cv2.putText(
    image,
    "Faces: " + str(count),
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

# Save output
output_path = os.path.join(folder, "face_count_result.jpg")
cv2.imwrite(output_path, image)

# Display result
cv2.imshow("Face Detection and Counting", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open output folder
os.startfile(folder)
