
import tensorflow as tf

def pretrain_vgg():
    vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(vgg.get_layer('block5_pool').output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    
    bbox_num = 5
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, kernel_initializer='he_normal')(x)
    
    x = tf.keras.layers.Dense(7*7*(20 + 5 * bbox_num), kernel_initializer='he_normal', activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((7, 7, 45))(x)
    
    # box_reg = tf.keras.layers.Conv2D(filters=5*bbox_num, kernel_size=(3, 3), 
    #                            strides=(1, 1), padding='same', 
    #                            kernel_initializer='he_normal',
    #                            activation='sigmoid')(x)
    # class_prob = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), 
    #                            strides=(1, 1), padding='same', 
    #                            kernel_initializer='he_normal',
    #                            activation='softmax')(x)
    # x = tf.keras.layers.concatenate([box_reg, class_prob], axis=3)
    yolo = tf.keras.Model(inputs=vgg.input, outputs=x)
    return yolo

def pretrained_yolo():
    x_input = tf.keras.layers.Input(shape = (448, 448, 3), name="0")
    
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), name="1",
                               strides=(2, 2), padding='same', 
                               kernel_initializer='he_normal')(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), name="2",
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', name='lastconv',
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1000, activation = 'softmax', kernel_initializer=tf.keras.initializers.he_normal())(x)
    yolo_pre = tf.keras.Model(inputs=x_input, outputs=x)
    return yolo_pre

def yolo_model():
    
    yolo = pretrained_yolo()
    #yolo.load_weights('./pretrained_yolo_voc.h5')
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(yolo.get_layer('lastconv').output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(2, 2), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    
    bbox_num = 5
    box_reg = tf.keras.layers.Conv2D(filters=5*bbox_num, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal',
                               activation='sigmoid')(x)
    class_prob = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), 
                               strides=(1, 1), padding='same', 
                               kernel_initializer='he_normal',
                               activation='softmax')(x)
    x = tf.keras.layers.concatenate([box_reg, class_prob], axis=3)
    yolo = tf.keras.Model(inputs=yolo.input, outputs=x)
    return yolo