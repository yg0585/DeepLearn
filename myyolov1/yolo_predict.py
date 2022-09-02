import tensorflow as tf
from model import pretrain_vgg
import cv2
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def get_test_img(cnts):
    imgs = []
    coordi = []
    cnt = 0
    image_size = 224
    with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_test.txt", "r") as f:
            for line in f:
                if cnt < cnts:
                    cnt += 1
                    continue
                if cnt == (cnts + 1):
                    break
                sline = line.split() 
                num_anc = len(sline) - 1
                file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
                img = cv2.imread(file)
                cnt += 1
                for i in range(num_anc):
                    coord = sline[i + 1].split(',')
                    xmin = float(coord[0]) * image_size / img.shape[1]
                    xmax = float(coord[2]) * image_size / img.shape[1]
                    ymin = float(coord[1]) * image_size / img.shape[0]
                    ymax = float(coord[3]) * image_size / img.shape[0]
                    a = [xmin, ymin, xmax, ymax]
                    coordi.append(a)
                img = (cv2.resize(img, (image_size, image_size)))
                imgs.append(img.tolist())
    return np.array(imgs), np.array(coordi)

def show_img(img, y_true, y_pred):
    lable_name = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                    "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    plt.imshow(img)
    ax = plt.gca()
    for i in range(y_true.shape[0]):
        rect = Rectangle((y_true[i][0], y_true[i][1]),
                        y_true[i][2] - y_true[i][0],
                        y_true[i][3] - y_true[i][1],
                    linewidth=2,
                    edgecolor='cyan',
                    fill = False) # 실제 답
    
        ax.add_patch(rect)

    idx = 0
    bbox_num = 5
    image_size = 224
    y_pred = np.array(y_pred)
    for i in range(7):
        for j in range(7):
            for k in range(bbox_num):
                conf_max = 0
                if y_pred[0, j, i, 4 + 5 * k] > conf_max:
                    conf_max = y_pred[0, j, i, 4 + 5 * k]
                    idx = k * 5
            print(conf_max, end=' ')
            if conf_max >= 0.4:
                x_cen = (j + y_pred[0, j, i, idx + 0]) * image_size / 7
                y_cen = (i + y_pred[0, j, i, idx + 1]) * image_size / 7
                w = y_pred[0, j, i, idx + 2] * image_size
                h = y_pred[0, j, i, idx + 3] * image_size
                xmin = x_cen - w/2
                ymin = y_cen - h/2
                #print("x_cen : ", x_cen, "y_cen : ", y_cen, "w : ", w, "h : ", h, 
                #      "pred_x : ", y_pred[0, j, k, 0], "pred_y : ", y_pred[0, j, k, 1])
                rect = Rectangle((xmin, ymin),
                            w,
                            h,
                        linewidth=2,
                        edgecolor='red',
                        fill = False)
                
                text = np.argmax(y_pred[0][j][i][bbox_num * 5:])
                ax.text(xmin, ymin, lable_name[text], fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
                ax.add_patch(rect)
        print()

    plt.show()


yolo = pretrain_vgg()
yolo.load_weights('./myyolo_bbox5_224_hasobj5_fc.h5')
#x_test, y_test = get_test_img(3)

for i in range(10):
    x_test, y_test = get_test_img(i)
    print(x_test.shape, y_test.shape)
    pred = yolo.predict(x_test)
    print(pred.shape)
    show_img(x_test[0], y_test, pred)