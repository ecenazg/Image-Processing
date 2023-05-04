#firstly import the libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Read a colored image and convert it into a gray-scale image and display the result
image = cv2.imread('resim.jpg')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray_img, cmap='gray')
plt.show()

if image is None:
    print('Could not read image')

    # Display the result using a colormap.
kernel1 = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
 
identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
plt.imshow(identity, cmap='cool')
plt.show()
#Filter the gray-scale image using each of the following filters
#Display the absolute value of colormap.
abs_img = np.abs(identity)
plt.imshow(abs_img, cmap='ocean') 
plt.show()

# Display the result using a colormap.
kernel2 = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
 
identity2 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
plt.imshow(identity2, cmap='pink')
plt.show() 
#Filter the gray-scale image using each of the following filters
#Display the absolute value of colormap.    
abs_img = np.abs(identity2)
plt.imshow(abs_img, cmap= 'pink')
plt.show()

