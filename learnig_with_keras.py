from learn.make_dataset import *
from learn.model.model_1 import my_model_1



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


if __name__ == "__main__":
    NOT_BALL_PATH = "ball_nasi/4-1"
    BALL_PATH = "ball_ari/ball1"
    dir_list = [NOT_BALL_PATH, BALL_PATH]
    NUM_CLASSES = 2
    X_train, X_test, y_train, y_test = make_dataset(dir_list, NUM_CLASSES)
    print(len(X_train))
    input_shape = X_train[0].shape
    model = my_model_1(X_train[0].shape, NUM_CLASSES)
    history = model.fit(X_train, y_train, batch_size=5, epochs=100, validation_data=(X_test, y_test), verbose = 1)
    print(model.evaluate(X_test, y_test))