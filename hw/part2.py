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

def gaussianSmoothing(img, sigma):
    ksize = int(6*sigma) + 1
    kernel = cv2.getGaussianKernel(ksize, sigma)
    smoothedImg = cv2.sepFilter2D(img, -1, kernel, kernel)
    return smoothedImg
sigmasforbackground=[5,10]
fig, axs = plt.subplots(1, len(sigmasforbackground)+1, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(bgImg, cv2.COLOR_BGR2RGB))
axs[0].set_title("Background Image")
for i, sigma in enumerate(sigmasforbackground):
    smoothedImg = gaussianSmoothing(bgImg, sigma)

    axs[i+1].imshow(cv2.cvtColor(smoothedImg,cv2.COLOR_BGR2RGB))
    axs[i+1].set_title(f"Sigma = {sigma}")
plt.show()

def unsharpMasking(img, sigma):
    blurredImg = gaussianSmoothing(img, sigma)
    unsharpImg = cv2.addWeighted(img, 1.5, blurredImg, -0.5, 0)
    return unsharpImg      
sigmasforforeground=[3,8]
fig, axs = plt.subplots(1, len(sigmasforforeground)+1, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(fgImg, cv2.COLOR_BGR2RGB))
axs[0].set_title("Foreground Image")
for i, sigma in enumerate(sigmasforforeground):
    smoothedImg = unsharpMasking(fgImg, sigma)
    axs[i+1].imshow(cv2.cvtColor(smoothedImg,cv2.COLOR_BGR2RGB))
    axs[i+1].set_title(f"Sigma = {sigma}")
plt.show()

def combineSharpenedForegroundSmoothedBackground(fgImg, fgMask, bgImg, bgSigma, fgSigma, topLeft):
    smoothedBgImg = gaussianSmoothing(bgImg, bgSigma)
    sharpenedFgImg = unsharpMasking(fgImg, fgSigma)
    combinedImg = combineForegroundBackground(sharpenedFgImg, fgMask, smoothedBgImg, topLeft)

    return combinedImg
combinedImg = combineSharpenedForegroundSmoothedBackground(fgImg, fgMask, bgImg, 0, 0, ((200, 400)))
combinedImg2 = combineSharpenedForegroundSmoothedBackground(fgImg, fgMask, bgImg, 3, 5, ((200, 400)))
combinedImg3 = combineSharpenedForegroundSmoothedBackground(fgImg, fgMask, bgImg, 8, 10, ((200, 400)))
fig, axs = plt.subplots(1, 3)
fig.set_size_inches(15,5)

axs[0].imshow(combinedImg[:,:,::-1])
axs[0].set_title("Combined Image 1")
axs[1].imshow(combinedImg2[:,:,::-1])
axs[1].set_title("Combined Image 2")
axs[2].imshow(combinedImg3[:,:,::-1])
axs[2].set_title("Combined Image 3")

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()