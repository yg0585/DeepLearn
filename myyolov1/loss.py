import tensorflow as tf
import numpy as np

def iou_score(true, pred, grid_shape=(7, 7)):
    grid_shape = np.array(grid_shape[::-1])
    xy_true = true[..., 0:2]/grid_shape
    wh_true = true[..., 2:4]
    
    xy_pred = pred[..., 0:2]/grid_shape
    wh_pred = pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true = xy_true - half_xy_true
    maxes_true = xy_true + half_xy_true
    
    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred  
    
    intersect_mins  = tf.maximum(mins_pred,  mins_true)
    intersect_maxes = tf.minimum(maxes_pred, maxes_true)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1] 
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1] 

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/union_areas
    
    # print("--------here is iou_scores----------")
    # tf.print(iou_scores, summarize=-1)
    return iou_scores

def get_loss(y_true, y_pred):
    # grid_shape =(7, 7)
    # xywhc_true = tf.reshape(y_true[..., 0:5], #N*7*7*1*5
    #                         (-1, *grid_shape, 1, 5))
    # xywhc_pred = tf.reshape(y_pred[..., 0:5], #N*7*7*1*5
    #                         (-1, *grid_shape, 1, 5))
    
    # xy_true = y_true[..., 0:2] #N*7*7*2
    # xy_pred = y_pred[..., 0:2] #N*7*7*2
    # xy_true = tf.expand_dims(xy_true, axis=-1) #N*7*7*2
    # xy_pred = tf.expand_dims(xy_pred, axis=-1) #N*7*7*2
    
    # iou = iou_score(xywhc_true, xywhc_pred) #N*7*7*1*1
    # iou = tf.one_hot(tf.argmax(iou, axis=-1),
    #                                depth=1,
    #                                dtype=xywhc_true.dtype) # N*S*S*B
    # iou = tf.expand_dims(iou, axis=-1)
    
    # xy_loss = tf.reduce_sum(
    #     tf.reduce_mean(xywhc_true[...,4]*iou*
    #     tf.square(xy_true - xy_pred),
    #     axis=0)) * 5
    
    # wh_true = y_true[..., 2:4]
    # wh_pred = y_pred[..., 2:4]
    # wh_true = tf.expand_dims(wh_true, axis=-1)
    # wh_pred = tf.expand_dims(wh_pred, axis=-1)
    
    # wh_loss = tf.reduce_sum(
    #     tf.reduce_mean(xywhc_true[...,4]*iou*
    #                    tf.square(
    #     tf.sqrt(wh_true) - tf.sqrt(wh_pred)), axis=0)) * 5
    
    # c_pred = y_pred[...,4]
    # c_pred = tf.expand_dims(c_pred, axis=-1)
    # has_obj_loss = tf.reduce_sum(
    #     tf.reduce_mean(xywhc_true[...,4]*iou*
    #                    tf.square(iou - c_pred),
    #                    axis=0))
    
    # no_obj = 1 - xywhc_true[...,4]*iou
    # no_obj_loss = tf.reduce_sum(tf.reduce_mean(no_obj*
    #                              tf.square(0 - c_pred), axis=0)) * 0.5
    
    # p_true = y_true[..., 5:]
    # p_pred = y_pred[..., 5:]
    # p_true = tf.expand_dims(p_true, axis=-1)
    # p_pred = tf.expand_dims(p_pred, axis=-1)
    
    # p_loss = tf.reduce_sum(tf.reduce_mean(xywhc_true[...,4]*iou*
    #                         tf.square(p_true - p_pred), axis=0)) * 5
    # loss = xy_loss + wh_loss + has_obj_loss + no_obj_loss + p_loss
    grid_shape = 7, 7
    class_num = 20
    bbox_num = 5
    epsilon = 1e-07
    loss_weight=[5, 5, 1, 1]
    
    xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) 
    xywhc_pred = tf.reshape(
        y_pred[..., :-class_num],
        (-1, *grid_shape, bbox_num, 5)) 

    iou_scores = iou_score(xywhc_true, xywhc_pred, grid_shape) 
    response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                depth=bbox_num,
                                dtype=xywhc_true.dtype) 
    response_mask_exp = tf.expand_dims(response_mask, axis=-1) 

    has_obj_mask = xywhc_true[..., 4] 
    has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) 
    no_obj_mask = 1 - has_obj_mask*response_mask 

    xy_true = xywhc_true[..., 0:2] 
    xy_pred = xywhc_pred[..., 0:2] 

    wh_true = tf.maximum(xywhc_true[..., 2:4], epsilon) 
    wh_pred = tf.maximum(xywhc_pred[..., 2:4], epsilon)

    c_pred = xywhc_pred[..., 4] 
    
    xy_loss = tf.reduce_sum(
        tf.reduce_mean(
            has_obj_mask_exp 
            *response_mask_exp 
            *tf.square(xy_true - xy_pred),
            axis=0))

    wh_loss = tf.reduce_sum(
        tf.reduce_mean(
            has_obj_mask_exp 
            *response_mask_exp 
            *tf.square(tf.sqrt(wh_true) - tf.sqrt(wh_pred)), 
            axis=0))

    has_obj_c_loss = tf.reduce_sum(
        tf.reduce_mean(
            has_obj_mask
            *response_mask 
            *tf.square(iou_scores - c_pred), 
            axis=0))

    no_obj_c_loss = tf.reduce_sum(
        tf.reduce_mean(
            no_obj_mask 
            *tf.square(0 - c_pred), 
            axis=0))
    
    c_loss = 50 * has_obj_c_loss + 0.5 * no_obj_c_loss

    p_true = y_true[..., -class_num:] 
    p_pred = y_pred[..., -class_num:] 
    p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)

    p_loss = -tf.reduce_sum(
        tf.reduce_mean(
            has_obj_mask 
            *p_true*tf.math.log(p_pred), 
            axis=0))
    loss = (loss_weight[0]*xy_loss
            + loss_weight[1]*wh_loss
            + loss_weight[2]*c_loss
            + loss_weight[3]*p_loss)
    return loss