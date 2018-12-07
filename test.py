from keras.engine.saving import load_model

from predict.normalize_img import *
from load.load_model import *
import os
from tqdm import tqdm

def test(img_path):
    img = cv2.imread(img_path)
    return img


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    print("1")
    ext = ".jpg"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(curDir, file)
                print("targetpaths is")
                print(img_path)
                print("-----------------------------")
                img_path_list.append(img_path)
    return img_path_list




if __name__ == '__main__':

    img_path_list = make_img_list("./ball_ari")

    label = {0: "ボールなし",
             1: "ボールあり"}
    # img_path = "/home/nakatsuka/PycharmProjects/ball_detection/ball_ari/ball_90/90_2x_285_y_210.jpg"
    model_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.json"
    weight_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.h5"

    model = load_json(model_path)
    model.load_weights(weight_path)


    for path in tqdm(img_path_list):
        test_img = test(path)
        img = normalize_img(test_img)
        predict = model.predict(img)

        print("識別結果 : ", label[predict.argmax()])

