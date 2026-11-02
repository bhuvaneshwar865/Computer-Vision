
import cv2
import os
import numpy as np
import urllib.request

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))

# Load eye cascade classifier
cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(cascade_path)

if eye_cascade.empty():
    print("Error: Eye classifier not found!")
    os.startfile(folder)
    exit()

# Image URL
url = "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_1200,q_60/lsci/db/PICTURES/CMS/96100/96159.jpg"

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

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw rectangles around eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Save output image
    output_path = os.path.join(folder, "eye_detection_result.jpg")
    cv2.imwrite(output_path, image)

    # Display result
    cv2.imshow("Human Eye Detection", image)
    print("Number of eyes detected:", len(eyes))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Eye detection completed!")
    os.startfile(folder)

except Exception as e:
    print("Error:", e)
    os.startfile(folder)
