import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image using cv2.imread
img = cv2.imread(r"C:\Users\ecena\OneDrive\Belgeler\Image Processing\rice.png", cv2.IMREAD_GRAYSCALE)

# Threshold the image using cv2.threshold
ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Define the kernel for dilation and erosion operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Apply dilation and erosion operations
dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
eroded_img = cv2.erode(thresh_img, kernel, iterations=1)

# Display the original image and binary image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(thresh_img, cmap='gray')
plt.title('Binary Threshold Image')
plt.xticks([]), plt.yticks([])

# Display the dilated and eroded images
plt.subplot(2, 2, 3)
plt.imshow(dilated_img, cmap='gray')
plt.title('Dilation Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(eroded_img, cmap='gray')
plt.title('Erosion Image')
plt.xticks([]), plt.yticks([])

plt.show()