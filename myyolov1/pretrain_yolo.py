import tensorflow as tf
from model import pretrained_yolo
import os
import cv2
import numpy as np

def get_data_imagenet():
    while True:
        path = "D:/imagenet2012/train"
        polder = os.listdir(path)
        idx = 0
        for pname in polder:
            sub_path = os.path.join(path, pname)
            file = os.listdir(sub_path)
            cnt = 0
            for fname in file:
                if cnt == 500 :
                    break
                img = cv2.imread(os.path.join(sub_path, fname))
                img = cv2.resize(img, (448, 448))
                label = np.array(idx)
                label = tf.keras.utils.to_categorical(label, 1000)
                label = label.astype(np.int8)
                cnt += 1

                yield np.expand_dims(img,axis=0), np.expand_dims(label, axis=0)
            idx += 1
            print("now label : ", idx)
def get_data_voc():
    while True:
        with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_train.txt", "r") as f:
            for line in f:
                sline = line.split() #파일 이름
                file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
                img = cv2.imread(file)
                img = (cv2.resize(img, (448, 448)))
                img = np.expand_dims(img, axis=0)
                coord = sline[1].split(',')
                label = np.array(coord[4])
                label = tf.keras.utils.to_categorical(label, 1000)
                label = label.astype(np.int8)
                yield img, np.expand_dims(label, axis=0)
pre_yolo = pretrained_yolo()
pre_yolo.summary()
train_feed = get_data_voc()
pre_yolo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    #optimizer=tf.keras.optimizers.SGD(momentum=0.9, decay=0.0005),
    loss="categorical_crossentropy",
    )

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./pretrained_yolo_voc.h5'   ,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=False)

#pre_yolo.load_weights('./pretrained_yolo.h5')
pre_yolo.fit(train_feed,
         steps_per_epoch=128,
         epochs=30000, 
         verbose=1,
         callbacks=[model_checkpoint_callback])