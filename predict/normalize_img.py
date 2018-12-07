import cv2
from keras.utils import np_utils
import numpy as np


def normalize_img(img):
    resized_img = cv2.resize(img, (30, 30))
    norm_img = np.array(resized_img)
    norm_img = norm_img.astype('float32')
    norm_img = np.expand_dims(norm_img, axis=0)
    norm_img = norm_img/255.0
    return norm_img


