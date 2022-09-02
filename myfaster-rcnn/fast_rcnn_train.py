from roi import get_roi_generator
from VGG16 import get_model
from anchor import fast_generator
import tensorflow as tf

def fast_cls_loss(*args):
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    # target = tf.gather_nd(y_true, indices)
    # output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.CategoricalCrossentropy()
    return lf(y_true, y_pred)

def fast_reg_loss(*args):
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(0.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.Huber()
    return lf(target, output)

def fast_rcnn():
    rpn_model, feature_map = get_model()
    rpn_cls_pred, rpn_reg_pred = rpn_model.output
    roi_pooled = get_roi_generator(feature_map.output, rpn_cls_pred, rpn_reg_pred)
    output = tf.keras.layers.Flatten()(roi_pooled)
    output = tf.keras.layers.Dense(4096, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.7)(output)   
    output = tf.keras.layers.Dense(4096, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.7)(output)
    cls_pred = tf.keras.layers.Dense(20, activation='softmax')(output)
    reg_pred = tf.keras.layers.Dense(20*4, activation='linear')(output)
    fast_model = tf.keras.Model(inputs=rpn_model.input, outputs=[cls_pred, reg_pred])
    return fast_model

def fast_rcnn_train():
    fast_model = fast_rcnn()
    fast_train_feed = fast_generator()
    fast_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5), 
                       loss=[fast_cls_loss, fast_reg_loss])
    fast_model.fit(fast_train_feed, stes_per_epoch = 256, epochs = 200, batch_size=16)

fast_rcnn_train()