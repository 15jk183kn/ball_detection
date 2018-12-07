import numpy as np
import cv2
import os
from tqdm import tqdm

# ルックアップテーブルの生成
def image_cont(img_path):
    min_table = 100
    max_table = 160
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    # 変換
    src = cv2.imread(img_path)
    high_cont_img = cv2.LUT(src, LUT_HC)
    low_cont_img = cv2.LUT(src, LUT_LC)
    return high_cont_img, low_cont_img
# cv2.imwrite("high_cont.png", high_cont_img)
# cv2.imwrite("low_cont.png", low_cont_img)



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
    img_path_list = make_img_list("ball_ari/ball")
    save_dir = "ball_ari/ball_cont100_160"
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
    for path in tqdm(img_path_list):
        high_filename = "high_" +os.path.basename(path)
        low_filename = "low_" + os.path.basename(path)
        high_img, low_img = image_cont(path)
        high_save_path = os.path.join(save_dir, high_filename)
        low_save_path = os.path.join(save_dir, low_filename)
        cv2.imwrite(high_save_path, high_img)
        cv2.imwrite(low_save_path, low_img)


