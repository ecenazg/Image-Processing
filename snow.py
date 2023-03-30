import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
path =r"C:\Users\ecena\Downloads\snow.jpg"
img= cv2.imread(path)

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# split channels
h, s, v = cv2.split(hsv)

# display V channel
cv2.imshow('V channel', v)

# display histogram of V channel
plt.hist(v.ravel(), 256, [0, 256])
plt.show()

# apply histogram equalization to V channel
v_eq = cv2.equalizeHist(v)

# display histogram of V channel after histogram equalization
plt.hist(v_eq.ravel(), 256, [0, 256])
plt.show()

# merge channels
hsv_eq = cv2.merge((h, s, v_eq))

# convert to RGB
img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# display final result
cv2.imshow('Final result', img_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()