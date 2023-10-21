import cv2
import numpy as np

# Load the image and mask
img = cv2.imread('n0323.jpg')
mask = cv2.imread('n0323.bmp', cv2.IMREAD_GRAYSCALE)
OD_mask = np.zeros_like(mask)
OC_mask = np.zeros_like(mask)
# Create OD_mask with all non-zero values set to 1
OD_mask[mask > 0] = 1

OC_mask[mask > 76] = 1

# Find contours in OD_mask and OC_mask
OD_contours, _ = cv2.findContours(OD_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
OC_contours, _ = cv2.findContours(OC_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
cv2.drawContours(img, OD_contours, -1, (255, 0,0 ), 2)  # green color for OD contours
cv2.drawContours(img, OC_contours, -1, (0, 100, 0), 2)  # red color for OC contours

# Save the resulting image
cv2.imwrite('mask_cover_img.jpg', img)