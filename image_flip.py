import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


def image_flip(img_path):
    img = cv2.imread(img_path)
    h, w, colors = img.shape
    size = (w, h)
    center = ((w-1) / 2, (h-1) / 2)
    print(center)
    angle = 270
    scale = 1.0
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(img, matrix, size)
    return img2

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# plt.show()


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
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


if __name__ == "__main__":
    img_path_list = make_img_list("./ball_ari/ball")
    save_dir = "ball_ari/ball_270"
    for path in tqdm(img_path_list):
        filename = "270_" +os.path.basename(path)
        flip_img = image_flip(path)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, flip_img)

