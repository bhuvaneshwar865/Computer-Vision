
import cv2
import numpy as np
import os

# Create white image
image = np.ones((500, 700, 3), dtype=np.uint8) * 255

# Display fixed text
cv2.putText(
    image,
    "WELCOME TO OPENCV",
    (60, 260),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 0, 255),
    3
)

# Get program folder
folder = os.path.dirname(os.path.abspath(__file__))

# Save output
cv2.imwrite(os.path.join(folder, "text_image.jpg"), image)

# Show output
cv2.imshow("Text on Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open output folder
os.startfile(folder)
