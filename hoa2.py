import cv2 as cv 
import numpy as np

path =r"C:\Users\ecena\Downloads\mitski.jpg"
image= cv.imread(path)
RbG_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#split the image into its three channels
(R,G,B) = cv.split(RbG_img)
#create named windows for each of the images we are going to display
cv.namedWindow("Blue", cv.WINDOW_NORMAL)
cv.namedWindow("Green", cv.WINDOW_NORMAL)
cv.namedWindow("Red", cv.WINDOW_NORMAL)
hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# split the HSV image into its H, S, and V channels
h, s, v = cv.split(hsv_img)
lab_img = cv.cvtColor(image, cv.COLOR_BGR2LAB)

# split the Lab image into its L, a, and b channels
l, a, b = cv.split(lab_img)

# display the L, a, and b channels separately
cv.imshow('L Channel', l)
cv.imshow('a Channel', a)
cv.imshow('b Channel', b)
# display the H, S, and V channels separately
cv.imshow('Hue Channel', h)
cv.imshow('Saturation Channel', s)
cv.imshow('Value Channel', v)
#display the images
cv.imshow("Blue",B)
cv.imshow("Green", G)
cv.imshow("Red", R)
if cv.waitKey(0):
    cv.destroyAllWindows()