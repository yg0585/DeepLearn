import numpy as np
import cv2

def xy_generator():
    while True:
        with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_train.txt", "r") as f:
            for line in f:
                sline = line.split() #파일 이름
                num_anc = len(sline) - 1
                file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
                img = cv2.imread(file)
                #print(sline[0])
                label = np.zeros((7, 7, 25), dtype=np.float32)
                for i in range(num_anc):
                    coord = sline[i + 1].split(',')
                    # xmin = float(coord[0]) * 7 / img.shape[1]
                    # xmax = float(coord[2]) * 7 / img.shape[1]
                    # ymin = float(coord[1]) * 7 / img.shape[0]
                    # ymax = float(coord[3]) * 7 / img.shape[0]
                    xmin = float(coord[0])
                    xmax = float(coord[2])
                    ymin = float(coord[1])
                    ymax = float(coord[3])
                    w = xmax - xmin
                    h = ymax - ymin
                    x_cen = xmin + (w / 2.0)
                    y_cen = ymin + (h / 2.0)
                    x_cen = x_cen * 7.0 / img.shape[1]
                    y_cen = y_cen * 7.0 / img.shape[0]
                    x_i = int(x_cen)
                    y_i = int(y_cen)
                    #print("name : ", sline[0], "center : ",x_cen, y_cen, "idx : ", x_i, y_i)
                    if x_i < 7 and y_i < 7:
                        label[y_i, x_i, 0] = x_cen - x_i
                        label[y_i, x_i, 1] = y_cen - y_i
                        label[y_i, x_i, 2] = w / img.shape[1]
                        label[y_i, x_i, 3] = h / img.shape[0]
                        label[y_i, x_i, 4] = 1.0
                        label[y_i, x_i, 5 + int(coord[4])] = 1.0
                img = (cv2.resize(img, (224, 224)))
                img = np.expand_dims(img, axis=0)
                yield img, label
