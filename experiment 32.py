import cv2
import numpy as np
import os

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))

# Get image dimensions from user
height = int(input("Enter image height: "))
width = int(input("Enter image width: "))

# Get box size
box_size = min(height, width) // 5

# Create white image
image = np.ones(
    (height, width, 3),
    dtype=np.uint8
) * 255

# Black box - Top Left
cv2.rectangle(
    image,
    (0, 0),
    (box_size, box_size),
    (0, 0, 0),
    -1
)

# Blue box - Top Right
cv2.rectangle(
    image,
    (width - box_size, 0),
    (width, box_size),
    (255, 0, 0),
    -1
)

# Green box - Bottom Left
cv2.rectangle(
    image,
    (0, height - box_size),
    (box_size, height),
    (0, 255, 0),
    -1
)

# Red box - Bottom Right
cv2.rectangle(
    image,
    (width - box_size, height - box_size),
    (width, height),
    (0, 0, 255),
    -1
)

# Save output image
cv2.imwrite(
    os.path.join(folder, "colored_boxes.jpg"),
    image
)

# Display image
cv2.imshow("Colored Boxes Image", image)

print("White image with four colored boxes created successfully.")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Open File Explorer
os.startfile(folder)
