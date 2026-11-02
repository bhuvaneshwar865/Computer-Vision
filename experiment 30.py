
import cv2
import os
import numpy as np
import urllib.request

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))

# Load smile cascade classifier
cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(cascade_path)

if smile_cascade.empty():
    print("Error: Smile classifier not found!")
    os.startfile(folder)
    exit()

# Image URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4fikOeote8mjCKla_BMf8kVsDweNygzThQWtIctZdgxMF2ecvm6r58nWe&s=10"

# Download image
request = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0"}
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("Error: Image not found!")
        os.startfile(folder)
        exit()

    # Convert image into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect smiles
    smiles = smile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(25, 25)
    )

    # Draw rectangles around detected smiles
    for (x, y, w, h) in smiles:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Save output image
    output_path = os.path.join(folder, "smile_detection_result.jpg")
    cv2.imwrite(output_path, image)

    # Display result
    cv2.imshow("Human Smile Detection", image)

    print("Number of smiles detected:", len(smiles))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Smile detection completed!")
    os.startfile(folder)

except Exception as e:
    print("Error:", e)
    os.startfile(folder)
