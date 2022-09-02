from cv2 import threshold
from data_util import generate_anchor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

#ymin, xmin, ymax, xmax
#xmin, ymin, xmax, ymax
def get_iou(bb1, bb2):
    
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    
    x_left = max(bb1[1], bb2[1])
    y_bottom = max(bb1[0], bb2[0])
    x_right = min(bb1[3], bb2[3])
    y_top = min(bb1[2], bb2[2])
    
    if x_right < x_left or y_bottom > y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_top - y_bottom)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_minibatch(anc_box, p_idx, n_idx, cd, cd_m):
    label = np.zeros((22500,), dtype=np.float32)
    label.fill(-1)
    random.shuffle(n_idx)
    # print("p_idx : ", len(p_idx))
    # print("n_idx : ", len(n_idx))
    # if (len(p_idx) == 1) :
    #     print(anc_box[p_idx])
    reg_label = np.zeros((22500, 4), dtype=np.float32)
    reg_label.fill(0)
    if len(p_idx) < 128 :
        i = 0
        for idx in p_idx:
            label[idx] = 1
            reg_label[idx] = cd[cd_m[i]]
            i += 1
        for i in range(256 - len(p_idx)):
            label[n_idx[i]] = 0
    else :
        random.shuffle(p_idx)
        r_idx = random.sample(range(len(p_idx)), 10)
        for i in range(128):
            label[n_idx[i]] = 0
        for i in range(128):
            label[p_idx[r_idx[i]]] = 1
            reg_label[p_idx[r_idx[i]]] = cd[cd_m[r_idx[i]]]
    label = np.reshape(label, (50, 50, 9))
    reg_label = np.reshape(reg_label, (50, 50, 36))
    return label, reg_label

def rpn_generator():
    anc_box, index_inside = generate_anchor()
    cnt = 0
    while True:
        with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_train.txt", "r") as f:
            for line in f:
                sline = line.split() #파일 이름
                num_anc = len(sline) - 1
                file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
                img = cv2.imread(file)
                img = (cv2.resize(img, (800, 800)))
                cnt += 1
                max = 0
                p_idx = []
                n_idx = []
                cd = []
                cd_m = []
                # print("file name : ", sline[0])
                for i in range(num_anc):
                    coord = sline[i + 1].split(',')
                    img_anc = np.zeros((4,), dtype=np.float32)
                    img_anc[1] = float(coord[0]) * 800 / img.shape[1]
                    img_anc[3] = float(coord[2]) * 800 / img.shape[1]
                    img_anc[0] = float(coord[1]) * 800 / img.shape[0]
                    img_anc[2] = float(coord[3]) * 800 / img.shape[0]
                    save_idx = 0
                    cd.append(img_anc)
                    for j in index_inside:
                        iou = get_iou(anc_box[j], img_anc)
                        if max < iou :
                            max = iou
                            save_idx = j
                            # print(iou , save_idx)
                        if iou >= 0.7 :
                            p_idx.append(j)
                            cd_m.append(i)
                        elif iou <= 0.3 :
                            n_idx.append(j)
                    p_idx.append(save_idx)
                    cd_m.append(i)
                # label.append(get_minibatch(p_idx, n_idx))
                label, reg_label = get_minibatch(anc_box, p_idx, n_idx, cd, cd_m)
                #print(sublabel)
                img = np.array(img)
                img = np.expand_dims(img, axis=0)
                yield img, (np.expand_dims(label, axis=0), np.expand_dims(reg_label, axis=0))
           
def fast_generator(): 
    anc_box, index_inside = generate_anchor()
    cnt = 0
    while True:
        with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_train.txt", "r") as f:
            for line in f:
                sline = line.split() #파일 이름
                num_anc = len(sline) - 1
                file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
                img = cv2.imread(file)
                img = (cv2.resize(img, (800, 800)))
                gt_label = np.zeros((20,))
                gt_boxes = np.zeros((20, 4), dtype=np.float32)
                cnt += 1
                max = 0
                p_idx = []
                n_idx = []
                cd = []
                cd_m = [] 
                # print("file name : ", sline[0])
                for i in range(num_anc):
                    coord = sline[i + 1].split(',')
                    gt_label[coord[5]] = 1
                    gt_boxes[coord[5]][1] = float(coord[0]) * 800 / img.shape[1]
                    gt_boxes[coord[5]][3] = float(coord[2]) * 800 / img.shape[1]
                    gt_boxes[coord[5]][0] = float(coord[1]) * 800 / img.shape[0]
                    gt_boxes[coord[5]][2] = float(coord[3]) * 800 / img.shape[0]
                    save_idx = 0
                    cd.append(gt_boxes[coord[5]])
                    for j in index_inside:
                        iou = get_iou(anc_box[j], gt_boxes[coord[5]])
                        if max < iou :
                            max = iou
                            save_idx = j
                            # print(iou , save_idx)
                        if iou >= 0.7 :
                            p_idx.append(j)
                            cd_m.append(i)
                        elif iou <= 0.3 :
                            n_idx.append(j)
                    p_idx.append(save_idx)
                    cd_m.append(i)
                # label.append(get_minibatch(p_idx, n_idx))
                label, reg_label = get_minibatch(anc_box, p_idx, n_idx, cd, cd_m)
                #print(sublabel)
                img = np.array(img)
                img = np.expand_dims(img, axis=0)
                yield (img, gt_boxes, gt_label, np.expand_dims(label, axis=0), np.expand_dims(reg_label, axis=0))
   
   