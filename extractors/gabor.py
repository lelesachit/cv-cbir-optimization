import cv2
import numpy as np


def gabor_extractor(image, num_orientations=8, num_scales=5):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize list to store Gabor filter features
    gabor_features = []

    # Define the range of orientations and scales for Gabor filters
    orientations = np.arange(0, np.pi, np.pi / num_orientations)
    scales = np.arange(0, num_scales)

    # Iterate over orientations and scales
    for theta in orientations:
        for scale in scales:
            # Create Gabor filter kernel
            kernel = cv2.getGaborKernel((5, 5), 2.0 ** scale, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)

            # Apply Gabor filter to the image
            filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)

            # Compute mean and standard deviation of filtered image as features
            mean = np.mean(filtered_image)
            std_dev = np.std(filtered_image)

            # Append features to list
            gabor_features.extend([mean, std_dev])

    return gabor_features