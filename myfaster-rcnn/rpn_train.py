from gc import callbacks
import tensorflow as tf
import cv2
import numpy as np

from VGG16 import get_model
from anchor import rpn_generator

def cls_loss(*args):
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)

def reg_loss(*args):
    # y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    # #
    # loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    # loss_for_all = loss_fn(y_true, y_pred)
    # loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    # #
    # pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    # pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    # #
    # loc_loss = tf.reduce_sum(pos_mask * loss_for_all)
    # total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(0.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.Huber()
    return lf(target, output)

def train_rpn():
    # imgs = []
    # with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_train.txt", "r") as f:
    #     for line in f:
    #         sline = line.split() #파일 이름
    #         num_anc = len(sline) - 1
    #         file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
    #         img = cv2.imread(file)
    #         imgs.append(cv2.resize(img, (800, 800)))
    # imgs = np.array(imgs)
    # print(imgs.shape)
    rpn_model, feature_map = get_model()
    rpn_train_feed = rpn_generator()
    rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5), loss=[cls_loss, reg_loss])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="./rpn.h5",
        monitor='loss',
        save_weights_only=True,
        save_freq='epoch',
        save_best_only=True
    )
    rpn_model.fit(rpn_train_feed, steps_per_epoch = 256, epochs= 200, callbacks=[checkpoint])
    # rpn_cls_output, rpn_reg_output = rpn_model.predict(imgs)
    # print(rpn_cls_output.shape, rpn_reg_output.shape)
    return 0

train_rpn()