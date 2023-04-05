import cv2
import numpy as np
import matplotlib.pyplot as plt

def createMask(img):
    # Create a rectangular bounding box around the foreground object
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)
    cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # Create a binary mask from the resulting mask
    mask = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')

    return mask

def combineForegroundBackground(fgImg, fgMask, bgImg, topLeft):
    # Get the dimensions of the foreground image and background image
    fgH, fgW = fgImg.shape[:2]
    bgH, bgW = bgImg.shape[:2]

    # Get the position of the top-left corner of the foreground image in the background image
    fgTop, fgLeft = topLeft

    # Calculate the position of the bottom-right corner of the foreground image in the background image
    fgBottom = fgTop + fgH
    fgRight = fgLeft + fgW

    # Create a copy of the background image
    outputImg = bgImg.copy()

    # Loop through each pixel in the foreground mask
    for y in range(fgH):
        for x in range(fgW):
            # If the pixel belongs to the foreground object
            if fgMask[y,x] > 0:
                # Calculate the corresponding position in the background image
                bgX = fgLeft + x
                bgY = fgTop + y

                # Make sure the position is within the bounds of the background image
                if bgX >= 0 and bgX < bgW and bgY >= 0 and bgY < bgH:
                    # Copy the pixel value from the foreground image to the background image
                    outputImg[bgY,bgX] = fgImg[y,x]

    return outputImg

# Load the foreground and background images
fgImg = cv2.imread(r"C:\Users\ecena\OneDrive\Belgeler\Image Processing\hw\image_2.jpg")
bgImg = cv2.imread(r"C:\Users\ecena\OneDrive\Belgeler\Image Processing\hw\background.jpg")

# Create a mask of the foreground image
fgMask = createMask(fgImg)

topLeft = (528, 279)
topMiddle = (150,1000)
topRight= (166, 456)


# Combine the foreground and background images
outputImg = combineForegroundBackground(fgImg, fgMask, bgImg, topLeft)
plt.imshow(outputImg[:,:,::-1])
plt.imshow(cv2.cvtColor(outputImg, cv2.COLOR_BGR2RGB))
plt.show()

#cv2.imshow("Output Image Left", outputImg)

outputImg2 = combineForegroundBackground(fgImg, fgMask, bgImg, topMiddle)
plt.imshow(outputImg2[:,:,::-1])
plt.imshow(cv2.cvtColor(outputImg2, cv2.COLOR_BGR2RGB))
plt.show()

outputImg3 = combineForegroundBackground(fgImg, fgMask, bgImg, topRight)
#cv2.imshow("Output Image Left", outputImg2)
plt.imshow(outputImg3[:,:,::-1])
plt.imshow(cv2.cvtColor(outputImg3, cv2.COLOR_BGR2RGB))
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()