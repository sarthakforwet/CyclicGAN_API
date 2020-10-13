# This Scirpt basically implements a U-Net architecture for the Generator model

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import config

kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
gamma_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

class GeneratorModel:
    def unet_architecture(self):
        inp = tf.keras.layers.Input(shape = (*config.IMAGE_SIZE))
        assert config.IMAGE_SIZE[0]==256, \
        "Image size {config.IMAGE_SIZE[:2]} not supported"

        downstack =[
            downsample(64, 4),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4)
        ]

        upstack = [
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4),
        ]

        out = tf.keras.layers.Conv2DTranspose(config.OUTPUT_CHANNELS, 4, strides=2,
        kernel_initializer=kernel_initializer, activation="tanh")

        x = inp
        skip_connections = []
        for down in downstack:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1][:-1]
        for up, skip_node in zip(upstack, skip_connections):
            x = up(x)
            x = tf.keras.layers.concatenate()([x, skip_node])

        x = out(x)
        model = tf.keras.Model(inputs=inp, outputs=x)

        return model

    def downsample(filters, size, apply_instancenorm=True):
        layer = tf.keras.Sequential()
        layer.add(tf.keras.layers.Conv2D(filters, size, strides=2, \
        padding="same", kernel_initializer=kernel_initializer, use_bias=False))

        if apply_instancenorm:
            layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer))

        layer.add(tf.keras.layers.LeakyReLU())
        return layer

    def upsample(filters, size, apply_dropout=False):
        layer = tf.keras.Sequential()
        layer.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
        padding="same", use_bias=True, kernel_initializer=kernel_initializer))

        layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer))

        if apply_dropout:
            layer.add(tf.keras.layers.Dropout(0.5))

        layer.add(tf.keras.layers.ReLU())
        return layer