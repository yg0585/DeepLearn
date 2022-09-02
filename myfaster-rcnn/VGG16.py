import tensorflow as tf

def get_model():
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(800, 800, 3))
    f_ext = base_model.get_layer("block5_conv3")
    output = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(f_ext.output)
    rpn_clsf = tf.keras.layers.Conv2D(9, (1, 1), activation='sigmoid')(output)
    rpn_reg = tf.keras.layers.Conv2D(9*4, (1, 1), activation='linear')(output)
    rpn_model = tf.keras.models.Model(inputs=base_model.input, outputs=[rpn_clsf, rpn_reg])
    return rpn_model, f_ext