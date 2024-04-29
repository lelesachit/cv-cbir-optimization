# preprocess.py

import cv2
import numpy as np
from preprocess import preprocessor

def blend_images(original_image):
    alpha = 0.6
    preprocessed_image = preprocessor(original_image)
    blended_image = cv2.addWeighted(original_image, alpha, preprocessed_image, 1 - alpha, 0)
    return blended_image
