import cv2
import numpy as np


def preprocessor(image):
    # Gradient sharpening
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    sharpened_image = np.uint8(gradient_magnitude)

    # Apply thresholding to sharpened image
    _, thresholded = cv2.threshold(sharpened_image, 100, 255, cv2.THRESH_BINARY)

    return thresholded
