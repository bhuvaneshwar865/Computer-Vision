
import cv2
import numpy as np
import os

# Create a white image
image = np.ones((500, 700, 3), dtype=np.uint8) * 255

# Draw a red rectangle
cv2.rectangle(image, (150, 100), (550, 400), (0, 0, 255), 3)

# Save output
folder = os.path.dirname(os.path.abspath(__file__))
cv2.imwrite(os.path.join(folder, "rectangle_image.jpg"), image)

# Display output
cv2.imshow("Rectangle Shape", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open output folder
os.startfile(folder)
