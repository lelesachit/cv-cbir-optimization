from skimage.feature import hog
from skimage import exposure
import cv2

def hog_extractor(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return list(fd)