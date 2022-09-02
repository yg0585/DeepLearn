from VGG16 import get_model
import cv2
import numpy as np
import tensorflow as tf

# def imgs_generator():
#     with open("C://Users/yg058/Desktop/study/DeepLearning/VOCdevkit/2007_train.txt", "r") as f:
#         for line in f:
#             sline = line.split() #파일 이름
#             file = "C://Users/yg058/Desktop/study/DeepLearning/" + sline[0]
#             img = cv2.imread(file)
#             img = (cv2.resize(img, (800, 800)))
#             img = np.array(img)
#             img = np.expand_dims(img, axis=0)
#             yield img

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_roi_generator(feature_map, cls_pred, reg_pred):
    # rpn_model, feature_map = get_model()
    # rpn_model.load_weights('./rpn.h5')
    # imgs = imgs_generator()
    # cls_pred, reg_pred = rpn_model.predict(imgs, verbose=1)
    # nms = []
    # box_indices = []
    # for i in range(32):
    #     boxes = tf.reshape(reg_pred[i], (22500, 4))
    #     # boxes = NormalizeData(boxes)
    #     scores = tf.reshape(cls_pred[i], (22500,))
    #     arr = tf.image.non_max_suppression(boxes, scores, 2000, 0.7)
    #     # nms.append(list(boxes[arr]))
    #     # box_indices.append(i)
    #     boxes = tf.linalg.normalize(
    #             boxes, ord='euclidean', axis=None, name=None
    #             )
    #     for j in range(arr.numpy()):
    #         nms.append(boxes[j])
    #         box_indices.append(i)
    batch_size=16
    pre_roi_bboxes = tf.reshape(reg_pred, (batch_size, 22500, 1 , 4))
    pre_roi_labels = tf.reshape(cls_pred, (batch_size, 22500, 1))
    nms, _, _, _ = tf.image.combined_non_max_suppression(
        pre_roi_bboxes,
        pre_roi_labels,
        max_output_size_per_class=2000,
        max_total_size=2000,
        iou_threshold=0.7,
    )
    roi_bboxes = tf.stop_gradient(nms)            
    batch_size, total_bboxes = tf.shape(roi_bboxes)[0], tf.shape(roi_bboxes)[1]
    #
    row_size = batch_size * total_bboxes
    # We need to arange bbox indices for each batch
    pooling_bbox_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), (1, total_bboxes))
    pooling_bbox_indices = tf.reshape(pooling_bbox_indices, (-1, ))
    pooling_bboxes = tf.reshape(roi_bboxes, (row_size, 4))
    # Crop to bounding box size then resize to pooling size
    pooling_feature_map = tf.image.crop_and_resize(
        feature_map,
        pooling_bboxes,
        pooling_bbox_indices,
        (7, 7)
    )
    final_pooling_feature_map = tf.reshape(pooling_feature_map, (batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2], pooling_feature_map.shape[3]))
    return final_pooling_feature_map
    # pooled = tf.image.crop_and_resize(
    #         feature_map.output,
    #         np.array(nms),
    #         np.array(box_indices),
    #         (7, 7)
    # )
    
    # return tf.reshape(pooled, (32, len(box_indices), 7, 7, 512))
    # ex0 = tf.image.crop_and_resize(
    #     feature_map.output,
    #     np.array([[72.9, 64.6, 119.5, 113.17]]),
    #     np.array([0]),
    #     # np.reshape(reg_pred[0], (22500, 4)),
    #     # nms[0],
    #     (7, 7)
    # )
    # ex = tf.image.crop_and_resize(
    #     feature_map.output,
    #     np.array([[72.9, 64.6, 119.5, 113.17]]),
    #     np.array([1]),
    #     # np.reshape(reg_pred[0], (22500, 4)),
    #     # nms[0],
    #     (7, 7)
    # )
