import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
img = cv2.imread(r"C:\Users\ecena\OneDrive\Belgeler\Image Processing\cute.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the FFT of the gray-scale image
fft = np.fft.fft2(gray)

# Shift the FFT to the center
fft_shift = np.fft.fftshift(fft)

# Compute the magnitude spectrum of the FFT
magnitude_spectrum = np.log(1 + np.abs(fft_shift))

# Visualize the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Shifted)')
plt.show()

# Inverse shift the FFT
fft_inverse_shift = np.fft.ifftshift(fft_shift)

# Convert back to the spatial domain using Inverse FFT
img_back = np.fft.ifft2(fft_inverse_shift)
img_back = np.real(img_back)

# Check if the image is the same
diff = np.abs(gray - img_back)
print("Max absolute difference: ", np.max(diff))

# Visualize the original image and the reconstructed image
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image')
plt.show()

# Convert the reconstructed image to the frequency domain using FFT
fft_reconstructed = np.fft.fft2(img_back)

# Shift the FFT to the center
fft_reconstructed_shift = np.fft.fftshift(fft_reconstructed)

# Compute the magnitude spectrum of the reconstructed FFT
magnitude_spectrum_reconstructed = np.log(1 + np.abs(fft_reconstructed_shift))

# Visualize the magnitude spectrum of the reconstructed FFT
plt.imshow(magnitude_spectrum_reconstructed, cmap='gray')
plt.title('Magnitude Spectrum of Reconstructed Image')
plt.show()

# Check if the reconstructed image is the same as the original image
diff_reconstructed = np.abs(fft_shift - fft_reconstructed_shift)
print("Max absolute difference (reconstructed): ", np.max(diff_reconstructed))

# Visualize the difference between the original FFT and the reconstructed FFT
plt.imshow(np.log(1 + np.abs(diff_reconstructed)))
plt.title('Difference between Original FFT and Reconstructed FFT')
plt.show()