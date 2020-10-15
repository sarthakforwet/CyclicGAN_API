import tensorflow as tf
from . import config

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def cyclic_loss(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image-cycled_image))

def identity_loss(real_image, same_image):
    return tf.reduce_mean(tf.abs(real_image-same_image))

def generator_loss(generated_op):
    gen_loss = loss(tf.ones_like(generated_op), generated_op)
    return gen_loss

def discriminator_loss(disc_real_op, disc_generated_op):
    real_loss = loss(tf.ones_like(disc_real_op), disc_real_op)
    gen_loss = loss(tf.zeros_like(disc_generated_op), disc_generated_op)
    return (real_loss + gen_loss) * config.REDUCTION_DISCRIMINATOR

