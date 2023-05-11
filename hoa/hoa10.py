import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread(r"C:\Users\ecena\OneDrive\Belgeler\Image Processing\Sample Inputs\flower.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection with two different sets of parameters
edges1 = cv2.Canny(gray, 50, 150)
edges2 = cv2.Canny(gray, 100, 150)

# Display the images using matplotlib
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# Show the original image
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')

# Show the edge-detected image with the first set of parameters
axs[1].imshow(edges1, cmap='gray')
axs[1].set_title('Canny Edge Detection (50, 150)')
axs[1].axis('off')

# Show the edge-detected image with the second set of parameters
axs[2].imshow(edges2, cmap='gray')
axs[2].set_title('Canny Edge Detection (100, 150)')
axs[2].axis('off')

plt.show()
