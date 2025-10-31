import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_pre

class ColorCalibration(tf.keras.layers.Layer):
    def __init__(self, reg_lambda=1e-4, **kw):
        super().__init__(**kw)                 # do NOT force dtype here
        self.reg_lambda = reg_lambda
        self.I = tf.constant(np.eye(3, dtype=np.float32))
    def build(self, input_shape):
        self.M = self.add_weight("ccm", shape=(3,3), dtype="float32",
                                 initializer=tf.keras.initializers.Identity(), trainable=True)
        self.b = self.add_weight("bias", shape=(3,), dtype="float32",
                                 initializer="zeros", trainable=True)
    def call(self, x):
        x32 = tf.cast(x, tf.float32)
        x_corr = tf.einsum("bhwc,cd->bhwd", x32, self.M) + self.b
        x_corr = tf.clip_by_value(x_corr, 0., 1.)
        self.add_loss(self.reg_lambda * tf.reduce_sum(tf.square(self.M - self.I)))
        return tf.cast(x_corr, x.dtype)

class ResNetV2Preprocess(tf.keras.layers.Layer):
    def call(self, x):
        z = tf.cast(x, tf.float32)
        return resnet_pre(z * 255.0)
