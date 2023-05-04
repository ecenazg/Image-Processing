import cv2
import numpy as np
from matplotlib import pyplot as plt

path =r"C:\Users\ecena\Downloads\snow.jpg"
img= cv2.imread(path)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

plt.subplot(221), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(v), plt.title('V channel')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.hist(v.ravel(), 256, [0, 256]), plt.title('V channel histogram')
plt.subplot(224), plt.hist(cv2.equalizeHist(v).ravel(), 256, [0, 256]), plt.title('V channel histogram after equalization')
plt.show()