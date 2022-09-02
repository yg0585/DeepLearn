from model import pretrain_vgg
from loss import get_loss
from tools import xy_generator
import tensorflow as tf

# def scheduler(epoch):
#     if epoch > 2930 :
#         return 0.0001
#     if epoch > 4102 :
#         return 0.0001
#     return 1e-10

yolo=pretrain_vgg()
yolo.summary()
yolo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    #optimizer=tf.keras.optimizers.SGD(momentum=0.9, decay=0.0005),
    loss=get_loss,
    )
 
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./myyolo_bbox5_224_hasobj50_fc.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
# train_feed = tf.data.Dataset.from_generator(xy_generator, 
#                                              (tf.int16, tf.float32),)
# train_feed = train_feed.batch(3)

#callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#yolo.load_weights('./myyolo_bbox5_224_hasobj50_fc.h5')
train_feed = xy_generator()
yolo.fit(train_feed,
         steps_per_epoch=128,
         epochs=3000, 
         verbose=1,
         callbacks=[model_checkpoint_callback])