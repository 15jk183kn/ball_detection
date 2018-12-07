from learn.make_dataset import *
from learn.model.model_1 import my_model_1
import cv2
from save_model import *
from datetime import datetime
import numpy as np


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".jpg"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(curDir, file)
                img_path_list.append(img_path)
    return img_path_list


def crate_save_as_time(save_root):
    tdatetime = datetime.now()
    tstr = tdatetime.strftime('%m%d-%a-%H')
    save_dir = os.path.join(save_root, tstr)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


if __name__ == "__main__":
    NOT_BALL_PATH = "ball_nasi"
    BALL_PATH = "ball_ari"
    NOT_BALL = 0
    BALL = 1
    NUM_CLASSES = 2
    X = []
    Y = []
    for path in make_img_list(NOT_BALL_PATH):
        img = cv2.imread(path)
        X.append(img)
        Y.append(NOT_BALL)

    for path in make_img_list(BALL_PATH):
        img = cv2.imread(path)
        X.append(img)
        Y.append(BALL)




    X, Y = normalize_dataset(X, Y, 2)


    k = 10
    num_val_samples = len(X) //k
    num_epochs = 100
    all_scores = []
    all_mae_histories = []
    acc = []
    loss = []
    for i in (range(k)):
        print('processing fold #', i)
        print(len(X))
        print(len(Y))
        print(X.shape)
        print(Y.shape)


        val_data = \
            X[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = \
            Y[i * num_val_samples: (i + 1) * num_val_samples]

        # print('X_i', X[:i * num_val_samples])
        # print('X_+i', X[(i + 1) * num_val_samples:])

        partial_train_data = np.concatenate(
            [X[:i * num_val_samples],
             X[(i + 1) * num_val_samples:]],
            axis = 0)
        #
        #
        # print(val_data)
        # print('=======================================')
        # print(val_targets)
        # print('=======================================')
        # print('Y_i', Y[:i * num_val_samples])
        # print('Y_i+1', Y[(i + 1) * num_val_samples:])
        # print('X_i', X[:i * num_val_samples])
        # print('X_+i', X[(i + 1) * num_val_samples:])

        partial_train_targets = np.concatenate(
            [Y[:i * num_val_samples],
             Y[(i + 1) * num_val_samples:]],
            axis = 0)

        model = my_model_1(X[0].shape, NUM_CLASSES)

        hist = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                  epochs = num_epochs, batch_size = 10, verbose = 0)

        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
        acc.append(val_mae)
        loss.append(val_mse)

        # mae_history = hist.history['val_mean_absolute_error']
        # all_mae_histories.append(mae_history)
        #
        # average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    print('acc:', acc)
    print('loss:', loss)
    print('val_mae:', np.average(acc))
    print('val_mse:', np.average(loss))


    # plt.plot(range(1, len() + 1), average_mae_history)
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation MAE')
    # plt.show()

    # X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(len(X_train))


    # input_shape = X_train[0].shape
    # model = my_model_1(X_train[0].shape, NUM_CLASSES)

    # batch_size = 16
    # epochs = 5
    # stack = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose = 1)

    # save_dir = crate_save_as_time("/home/nakatsuka/PycharmProjects/ball_detection/save")
    # save_model(save_dir, model)
    # save_weights(save_dir, model)
    # print(model.evaluate(X_test, y_test))



    # x = range(epochs)
    # plt.plot(x, stack.history['acc'], label="train_acc", color='blue')
    # plt.plot(x, stack.history['val_acc'], label="test_acc", color='red')
    # plt.title("accuracy")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    #
    # plt.plot(x, stack.history['loss'], label="train_loss", color='blue')
    # plt.plot(x, stack.history['val_loss'], label="test_loss", color='red')
    # plt.title("loss")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()


    #
    # model_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.json"
    # weight_path = "/home/nakatsuka/PycharmProjects/ball_detection/save/1126-Mon-16/<keras.engine.sequential.Sequential object at 0x7fe520081940>.h5"
    #
    #
    # model = load_json(model_path)
    # model.load_weights(weight_path)
    # predict_ball_ari =0
    # print(X_test.shape)
    # predict_list = model.predict(X_test)
    # ball_ari = 0
    # ball_nasi = 0
    # predict_ball_nasi = 0
    # count = 0
    # miss = 0
    # for predict_label, answer_label in zip(predict_list, y_test):
    #     if answer_label.argmax() == 1:
    #         ball_ari += 1
    #
    #     if answer_label.argmax() == 0:
    #         ball_nasi += 1
    #
    #     if (predict_label.argmax() == 1) and (answer_label.argmax() == 1):
    #         predict_ball_ari += 1
    #
    #     if (predict_label.argmax() == 0) and (answer_label.argmax() == 1):
    #         miss += 1
    #         print('Answer:', answer_label.argmax(), '- Predict', predict_label.argmax())
    #
    #     if (predict_label.argmax() == 0) and (answer_label.argmax() == 0):
    #         predict_ball_nasi +=1
    #
    #     if (predict_label.argmax() == 1) and (answer_label.argmax() == 0):
    #         count += 1
    #         print('Answer:', answer_label.argmax(), '- Predict', predict_label.argmax())
    #
    # cor = (predict_ball_ari + predict_ball_nasi) / (ball_ari + ball_nasi)
    # acc = predict_ball_ari / (predict_ball_ari + count)
    # det = predict_ball_ari / (predict_ball_ari + miss)
    # fva = 2 * acc * det / (acc + det)
    #
    # print('ボールあり画像：', ball_ari, '枚')
    # print('ボールありと正しく予測した画像：(TP)', predict_ball_ari, '枚')
    # print('ボールありなのにボールなしと予測した画像：(FN)', miss, '枚')
    # print('ボールなし画像：', ball_nasi, '枚')
    # print('ボールなしを正しく予測した画像：(TN)', predict_ball_nasi)
    # print('ボールなしなのにボールありと予測した画像：(FP)', count)
    # print('正解率:', cor)
    # print('精度:', acc)
    # print('検出率:', det)
    # print('F値:', fva)
