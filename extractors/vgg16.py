import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as img_preprocess
import cv2

def vgg16_preprocess(image):
    # Load pre-trained VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    img = cv2.resize(image, (224, 224))
    x = img_preprocess.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract features
    features = vgg.predict(x)

    return list(features.ravel())