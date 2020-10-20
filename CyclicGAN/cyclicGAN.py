import tensorflow as tf
from . import generator
from . import discriminator
from . import loss
from . import config

cfg = config.Configuration()
GeneratorModel = generator.GeneratorModel()
DiscriminatorModel = discriminator.DiscriminatorModel()
generator_loss = loss.generator_loss
discriminator_loss = loss.discriminator_loss
cyclic_loss = loss.cyclic_loss
identity_loss = loss.identity_loss

class CyclicGAN(tf.keras.Model):
    """Class to build and train custom CyclicGAN architecture."""
    def __init__(self):
        super(CyclicGAN, self).__init__()
        self.gen_g = GeneratorModel.unet_architecture()
        self.gen_f = GeneratorModel.unet_architecture()
        self.disc_x = DiscriminatorModel.PatchGAN()
        self.disc_y = DiscriminatorModel.PatchGAN()
        self.lambda_ = cfg.LAMBDA_CYCLIC

    def compile(self,
                gen_g_optimizer,
                gen_f_optimizer,
                disc_x_optimizer,
                disc_y_optimizer
               ):
        """Function to set the optimizers and metrics used for the model training."""
        super(CyclicGAN, self).compile()
        self.gen_g_optimizer= gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.gen_loss = generator_loss
        self.disc_loss = discriminator_loss
        self.cyclic_loss = cyclic_loss
        self.identity_loss = identity_loss

    def train_step(self, batch_data):
        """Function to run a single step of training."""
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:

            # Getting Generator and Discriminator output.
            fake_y = self.gen_g(real_x, training=True)
            cycled_x = self.gen_f(fake_y, training=True)

            fake_x = self.gen_f(real_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)

            same_y = self.gen_g(real_y, training=True)
            same_x = self.gen_f(real_x, training=True)

            disc_real_y = self.disc_y(real_y, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            disc_real_x = self.disc_x(real_x, training=True)
            disc_fake_x = self.disc_x(fake_x, training=True)

            # Calculate Losses
            cycle_loss = self.lambda_*(self.cyclic_loss(real_y, cycled_y, 10)+self.cyclic_loss(real_x, cycled_x, 10))
            identity_loss = self.lambda_*(self.identity_loss(real_y, same_y, 10) + self.identity_loss(real_x, same_x, 10))

            gen_g_loss = self.gen_loss(disc_fake_y)
            gen_f_loss = self.gen_loss(disc_fake_x)

            total_gen_g_loss = gen_g_loss + cycle_loss + identity_loss
            total_gen_f_loss = gen_f_loss + cycle_loss + identity_loss

            disc_y_loss = self.disc_loss(disc_real_y, disc_fake_y)
            disc_x_loss = self.disc_loss(disc_real_x, disc_fake_x)

        # Calculate Gradients
        gen_g_gradient = tape.gradient(total_gen_g_loss, self.gen_g.trainable_variables)
        gen_f_gradient = tape.gradient(total_gen_f_loss, self.gen_f.trainable_variables)

        disc_y_gradient = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)
        disc_x_gradient = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)

        self.gen_g_optimizer.apply_gradients(zip(gen_g_gradient, self.gen_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(gen_f_gradient, self.gen_f.trainable_variables))

        # Apply Gradients
        self.disc_y_optimizer.apply_gradients(zip(disc_y_gradient, self.disc_y.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(disc_x_gradient, self.disc_x.trainable_variables))

        return {
            "generator_g_loss": total_gen_g_loss,
            "discriminator_y_loss": disc_y_loss,
            "generator_f_loss": total_gen_f_loss,
            "discriminator_x_loss": disc_x_loss
        }
