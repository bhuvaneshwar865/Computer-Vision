
import cv2
import numpy as np
import os

# Create a white image
image = np.ones((500, 700, 3), dtype=np.uint8) * 255

# Draw a blue circle at the center
cv2.circle(image, (350, 250), 120, (255, 0, 0), 3)

# Get program folder
folder = os.path.dirname(os.path.abspath(__file__))

# Save output image
cv2.imwrite(os.path.join(folder, "circle_image.jpg"), image)

# Display output
cv2.imshow("Circle Shape", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open output folder
os.startfile(folder)
