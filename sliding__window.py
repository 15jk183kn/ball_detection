import cv2
import os

def sliding_window(img_path, save_dir, winW, winH, stepsize, dirname):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for y in range(0, h, stepsize):
        for x in range(0, w, stepsize):
            trim_img = img[y: y + winH, x: x + winW]
            filename = str(dirname) + "x_" + str(x)  +"_" + "y_" + str(y) + ".jpg"
            save_path = os.path.join(save_dir, filename)
            w1, h1, _ = trim_img.shape
            if (w1, h1) == (winW, winH):
                cv2.imwrite(save_path, trim_img)
            # cv2.imwrite(save_path, trim_img)



def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".png"
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
    img_path_list = make_img_list("./cap_KB57_pitcher_0001")
    for path in img_path_list:
        print(path)
        dirname = os.path.basename(path)
        dirname = dirname.replace(".png", "")
        save_dir = os.path.join("./trimmed_imgs2", dirname)
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        sliding_window(path, save_dir, 30, 30, 15, dirname)