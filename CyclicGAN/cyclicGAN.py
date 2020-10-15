import tensorflow as tf
from . import generator
from . import discriminator
from . import loss
from . import config

cfg = config.Configuration()
GeneratorModel = generator.GeneratorModel
DiscriminatorModel = discriminator.DiscriminatorModel
generator_loss = loss.generator_loss
discriminator_loss = loss.discriminator_loss
cyclic_loss = loss.cyclic_loss
identity_loss = loss.identity_loss

class CyclicGAN(tf.keras.Model):
    def __init__(self):
        super(CyclicGAN, self).__init__()
        gen_model = GeneratorModel()
        self.gen_xy = gen_model.unet_architecture()
        self.gen_yx = gen_model.unet_architecture()
        self.disc_x = DiscriminatorModel().PatchGAN()
        self.disc_y = DiscriminatorModel().PatchGAN()

    def compile(self, gen_xy_optim, gen_yx_optim, disc_x_optim, disc_y_optim):
        super(CyclicGAN, self).compile()
        self.gen_xy_optim = gen_xy_optim
        self.gen_yx_optim = gen_yx_optim
        self.disc_x_optim = disc_x_optim
        self.disc_y_optim = disc_y_optim

    def train_step(self, batch_data):
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            # Calculating outputs
            fake_x = self.gen_xy(real_y, training=True)
            cycled_y = self.gen_yx(fake_x, training=True)

            fake_y = self.gen_yx(real_x, training=True)
            cycled_x = self.gen_xy(fake_y, training=True)

            same_y = self.gen_yx(real_y, training=True)
            same_x = self.gen_xy(real_x, training=True)

            disc_real_x = self.disc_x(real_x, training=True)
            disc_fake_x = self.disc_x(fake_x, training=True)

            disc_real_y = self.disc_y(real_y, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            # Calculating loss
            cycle_loss = cfg.LAMBDA_CYCLIC*(cyclic_loss(real_x, cycled_x)+cyclic_loss(real_y, cycled_y))

            gen_yx_loss = generator_loss(disc_fake_y) + cycle_loss + cfg.LAMBDA_CYCLIC*identity_loss(real_y, same_y)
            gen_xy_loss = generator_loss(disc_fake_x) + cycle_loss + cfg.LAMBDA_CYCLIC*identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

            # Calculating gradient
            gen_xy_grad = tape.gradient(gen_xy_loss, self.gen_xy.trainable_variables)
            gen_yx_grad = tape.gradient(gen_yx_loss, self.gen_yx.trainable_variables)
            disc_x_grad = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
            disc_y_grad = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)

            # Updating the weights
            self.gen_xy_optim.apply_gradients(zip(gen_xy_grad, self.gen_xy.trainable_variables))
            self.gen_yx_optim.apply_gradients(zip(gen_yx_grad, self.gen_yx.trainable_variables))
            self.disc_x_optim.apply_gradients(zip(disc_x_grad, self.disc_x.trainable_variables))
            self.disc_y_optim.apply_gradients(zip(disc_y_grad, self.disc_y.trainable_variables))