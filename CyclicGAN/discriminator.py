import config
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

kernel_initializer = tf.random_normal_initializer(mean=0.0 stddev=0.02)
gamma_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

class DiscriminatorModel:
    def __init__(self):
        inp = tf.keras.layers.Input(shape=(*config.IMAGE_SIZE))

        x = downsample(64, 4, False)(inp)
        x = downsample(128, 4)(x)
        x = downsample(256, 4)(x)
        
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(x)
        conv1 = tf.keras.layers.Conv2D(512, 4, strides=1, padding="same"
        kernel_initializer=kernel_initializer, use_bias=False)(zero_pad1)

        norm_layer = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(conv1)
        activation_layer = tf.keras.layers.LeakyReLU()(norm_layer)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(activation_layer)

        out = tf.keras.layers.Conv2D(1, 4, strides=1, padding="same", kernel_initializer=kernel_initializer)(zero_pad2)
        model = tf.keras.Model(inputs=inp, outputs=out)
        return model

    def downsample(self, filters, size, apply_instancenorm=True):
        layer = tf.keras.Sequential()
        layer.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same",
        kernel_initializer=kernel_initializer, use_bias=False))

        if apply_instancenorm:
            layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer))
        
        layer.add(tf.keras.layers.LeakyReLU())

        return layer