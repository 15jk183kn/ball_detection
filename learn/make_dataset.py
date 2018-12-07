import glob
import os
import cv2
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def get_dataset(dir_list):
    X = []
    Y = []
    for label_num, dir in enumerate(dir_list):
        for path in sorted(glob.glob(dir + "/*.jpg")):
            print(path)
            img = cv2.imread(path)
            # 画像のタプルがこれになってなかったら弾く
            # スライディングウィンドウが間違っていたから
            if img.shape != (30, 30, 3):
                continue
            X.append(img)
            Y.append(label_num)
        print("{} is over".format(label_num))
    return X, Y



def normalize_dataset(X, Y, NUMCLASSES):
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype('float32')
    X = X/255.0
    Y = np_utils.to_categorical(Y, NUMCLASSES)
    return X, Y



def make_dataset(dir_root_list, NUMCLASSES):
    X, Y = get_dataset(dir_root_list)
    X, Y = normalize_dataset(X, Y, NUMCLASSES)
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#     return X_train, X_test, y_train, y_test
    return X, Y