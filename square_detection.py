import cv2
import os
from predict.normalize_img import *
from load.load_model import *


def square_detection():
    img_path = ("test_img/cap_1day_USA_Austraria_CAM4_0001_0002_FrameDIAS/4-6.png")
    img = cv2.imread(img_path)
    dirname = os.path.basename(img_path)
    dirname = dirname.replace(".png", "")
    save_dir = os.path.join("./trimed_img", dirname)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    winH = 20
    winW = 20
    stepsize = 10
    h, w, _ = img.shape



    model_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.json"
    weight_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.h5"

    model = load_json(model_path)
    model.load_weights(weight_path)

    label = {0: "ボールなし",
             1: "ボールあり"}

    file_list = []
    pre_list = []
    ari_list = []
    x_list = []
    y_list = []

    for y in range(0, h, stepsize):
        for x in range(0, w, stepsize):
            trim_img = img[y: y + winH, x: x + winW]
            filename = str(dirname) + "x_" + str(x)  +"_" + "y_" + str(y) + ".jpg"
            save_path = os.path.join(save_dir, filename)
            w1, h1, _ = trim_img.shape
            print(w1, h1)




            if (w1, h1) == (winW, winH):
                #cv2.imwrite(save_path, trim_img)
                #img2 = cv2.imread(save_path)
                img2 = normalize_img(trim_img)
                predict = model.predict(img2)
                print("識別結果 : ", label[predict.argmax()], filename,  predict.argmax())

                if predict.argmax() == 1:
                    print(filename)
                    file_list.append(filename)
                    # coordinate = (str(x), str(y))
                    # print(coordinate)
                    pre = (predict[()])
                    print(pre[0])
                    pre_list.append(pre[0][0])
                    pre_list.append(pre[0][1])
                    print(pre[0][1])
                    ari_list.append(pre[0][1])
                    x_list.append(x)
                    y_list.append(y)
                    # img2 = cv2.rectangle(img, (x, y), (x + 30, y + 30), (255, 0, 0))
                    # cv2.imshow("color", img2)
                    # cv2.imwrite("4-4s.png", img2)
                    dirname2 = "test_img"
                    filename2 = "square" + str(dirname) + ".jpg"
                    # print(filename2)
                    if not os.path.exists(dirname2):
                        os.mkdir(dirname)
                    cv2.imwrite(os.path.join(dirname2, filename2), img2)

    print('filename:', file_list)
    print('x_list:', x_list)
    print('y_list:', y_list)
    print('score:', pre_list)
    print('label1_score:', ari_list)


    print('正解インデックス：', np.argmax(ari_list))
    print('正解データ：',  np.max(ari_list))
    print('正解ファイル：', file_list[np.argmax(ari_list)])
    #
    #
    # true_filename = file_list[np.argmax(ari_list)]
    # true_x = x_list[np.argmax(ari_list)]
    # true_y = y_list[np.argmax(ari_list)]
    #
    # print('true_x:', true_x)
    # print('true_y:', true_y)
    #
    #
    #
    # img2 = cv2.rectangle(img, (true_x, true_y), (true_x + 30, true_y + 30), (255, 0, 0))
    # # cv2.imshow("color", img2)
    # # cv2.imwrite("4-4s.png", img2)
    # dirname2 = ("test_img")
    # filename2 = "square_" + str(dirname) + ".jpg"
    # # print(filename2)
    # if not os.path.exists(dirname2):
    #     os.mkdir(dirname)
    # cv2.imwrite(os.path.join(dirname2, filename2), img2)
    #
    # xx_list = []
    # yy_list = []
    #
    # for i in (x_list):
    #     print(i)
    #     # x = x_list[i]
    #     if ((true_x - winW) <= i <= (true_x + winH)):
    #         print(i)


if __name__ == '__main__':
    square_detection()