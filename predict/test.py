from predict.normalize_img import *
from load.load_model import *
import os

if __name__ == '__main__':
    #
    img_path = "/home/nakatsuka/PycharmProjects/ball_detection/ball_nasi/trimmed_imgscal/1/1x_315_y_180.jpg"
    model_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.json"
    weight_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.h5"

    model = load_json(model_path)
    model.load_weights(weight_path)

    label = {0: "ボールなし",
             1: "ボールあり"}
    img = cv2.imread(img_path)
    img = normalize_img(img)

    predict = model.predict(img)

    a = (predict[()])

    print(a[0])
    print(predict, type(predict), predict.shape)
    print(a[0][0], round(a[0][1], 3))
    # print("label[0,1] : ", (predict[()]))
    print("識別結果 : ", label[predict.argmax()])


    #
    # root = "/home/nakatsuka/PycharmProjects/ball_detection/ball_ari/ball"
    # for img_name in os.listdir(root):
    #     label = {0: "ボールなし",
    #              1: "ボールあり"}
    #     img_path = root + '/' + img_name
    #
    #     img = cv2.imread(img_path)
    #     img = normalize_img(img)
    #
    #     predict = model.predict(img)
    #
    #     print("識別結果 : ", label[predict.argmax()])
    #
