
import numpy as np
import cv2

def rgb_to_hsv(img):
    img = img.astype(np.float32) / 255.0
    red, green, blue = cv2.split(img)

    max_val = np.maximum(np.maximum(red, green), blue)
    min_val = np.minimum(np.minimum(red, green), blue)

    hue = np.zeros_like(max_val)

    # Compute hue
    idx = np.where(max_val == min_val)
    hue[idx] = 0.0

    idx = np.where((max_val == red) & (green <= blue))
    hue[idx] = 60.0 * ((green[idx] - blue[idx]) / (max_val[idx] - min_val[idx])) % 360.0

    idx = np.where((max_val == red) & (green > blue))
    hue[idx] = 60.0 * ((green[idx] - blue[idx]) / (max_val[idx] - min_val[idx])) + 360.0

    idx = np.where(max_val == green)
    hue[idx] = 60.0 * ((blue[idx] - red[idx]) / (max_val[idx] - min_val[idx])) + 120.0

    idx = np.where(max_val == blue)
    hue[idx] = 60.0 * ((red[idx] - green[idx]) / (max_val[idx] - min_val[idx])) + 240.0

    # Compute saturation
    saturation = np.where(max_val == 0.0, 0.0, (max_val - min_val) / max_val)

    # Compute value
    value = max_val

    hsv_img = cv2.merge([hue, saturation, value])
    hsv_img[...,0] /= 2
    hsv_img[...,0] = hsv_img[...,0].astype(np.uint8)
    hsv_img[...,1:] *= 255
    hsv_img = hsv_img.astype(np.uint8)

    return hsv_img

