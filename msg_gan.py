import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
import os
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import gan_layers
import gan_util as util
import gan_params as gp

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


# Uses reference code from
# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
class MsgGan:
    def __init__(self, target_res):
        assert np.ceil(np.log2(target_res)) == np.floor(np.log2(target_res)), 'Target resolution must be a power of 2'
        self.target_res = int(np.log2(target_res / 2))

        self.gen_opt = optimizers.Adam(learning_rate=gp.learning_rate, beta_1=gp.beta_1,
                                       beta_2=gp.beta_2, epsilon=gp.eps)
        self.crt_opt = optimizers.Adam(learning_rate=gp.learning_rate, beta_1=gp.beta_1,
                                       beta_2=gp.beta_2, epsilon=gp.eps)

        if gp.use_mixed_precision:
            self.gen_opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.gen_opt)
            self.crt_opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.crt_opt)

        # Build models
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Seed for keeping track of progress
        self.seed = util.generate_latents()

        # Set up log directory
        self.log_dir = util.init_log_dir()
        self.imgs_dir = util.init_imgs_dir()

        # Set up TensorBoard logging
        self.log_writer = tf.summary.create_file_writer(self.log_dir)
        self.total_seen = 0  # keep track of total number of images seen
        self.total_batches = 0  # keep track of total batches sen

        # Write model summaries to TensorBoard for debugging
        self.write_model_summaries()

        # Progress bar format
        self.pb_widgets = [
            '[', pb.Variable('batches', format='{formatted_value}'), '] ', pb.Variable('g_loss', precision=10), ', ',
            pb.Variable('c_loss', precision=10), ' [', pb.SimpleProgress(), '] ',
            pb.FileTransferSpeed(unit='img', prefixes=['', 'k', 'm']), '\t', pb.ETA()
        ]

    def random_image(self):
        r_img = self.generator(self.seed, training=False)
        self.view_imgs(r_img)

    def write_model_summaries(self):
        with self.log_writer.as_default():
            tf.summary.text('generator', self.generator.to_json(), step=0)
            tf.summary.text('critic', self.critic.to_json(), step=0)

    # view multi-scale generator output
    def view_imgs(self, imgs):
        # Get first image in batch
        imgs = [img[0] for img in imgs]
        fig, axs = plt.subplots(ncols=self.target_res)
        assert len(imgs) == self.target_res

        for subplot, img in enumerate(imgs):
            img = (img - np.min(img)) / np.ptp(img)  # Scale images to [0, 1]
            axs[subplot].axis('off')
            axs[subplot].imshow(img)
        plt.savefig(os.path.join(self.imgs_dir, str(self.total_seen) + '.png'))
        plt.cla()
        plt.close(fig)

    def train(self):
        epochs = 5

        dataset = util.load_celeba(self.target_res)
        for epoch in range(epochs):
            epoch_seen = 0
            with pb.ProgressBar(widgets=self.pb_widgets, max_value=gp.images_per_epoch) as progress:
                while epoch_seen < gp.images_per_epoch:
                    # Get a new batch of real images and create new generate input
                    real_batch = dataset.next()
                    latent_input = util.generate_latents()

                    critic_loss = self.train_critic(real_batch, latent_input)
                    generator_loss = self.train_generator(real_batch, latent_input)

                    epoch_seen += gp.batch_size
                    self.total_seen += gp.batch_size
                    self.total_batches += 1

                    # Write results to TensorBoard
                    with self.log_writer.as_default():
                        tf.summary.scalar('g_loss', generator_loss, step=self.total_seen)
                        tf.summary.scalar('c_loss', critic_loss, step=self.total_seen)

                    if util.time_to_update(epoch_seen):
                        progress.update(epoch_seen, batches=self.total_batches,
                                        g_loss=generator_loss, c_loss=critic_loss)

                    if util.time_for_img(epoch_seen):
                        self.random_image()

    # Hypothesis: getting the grads of the MEAN hinge loss loses batch data and destabilizes the training process
    # Relativistic Hinge loss (RaHinge)
    # See SAGAN paper: https://arxiv.org/pdf/1805.08318.pdf
    @tf.function
    def train_generator(self, real_batch, latent_input):
        with tf.GradientTape() as gen_tape:
            fake_batch = self.generator(latent_input, training=True)

            real_fake_diff = self.critic(real_batch, training=True) - tf.reduce_mean(self.critic(fake_batch, training=True))
            fake_real_diff = self.critic(fake_batch, training=True) - tf.reduce_mean(self.critic(real_batch, training=True))

            loss = tf.reduce_mean(tf.nn.relu(1 + real_fake_diff)) + tf.reduce_mean(tf.nn.relu(1 - fake_real_diff))
            scaled_loss = self.gen_opt.get_scaled_loss(loss)

        gen_vars = self.generator.trainable_variables
        scaled_gen_grads = gen_tape.gradient(scaled_loss, gen_vars)
        gen_grads = self.gen_opt.get_unscaled_gradients(scaled_gen_grads)
        self.gen_opt.apply_gradients(zip(gen_grads, gen_vars))
        return loss

    @tf.function
    def train_critic(self, real_batch, latent_input):
        with tf.GradientTape() as crt_tape:
            fake_batch = self.generator(latent_input, training=True)

            real_fake_diff = self.critic(real_batch, training=True) - tf.reduce_mean(self.critic(fake_batch, training=True))
            fake_real_diff = self.critic(fake_batch, training=True) - tf.reduce_mean(self.critic(real_batch, training=True))

            loss = tf.reduce_mean(tf.nn.relu(1 - real_fake_diff)) + tf.reduce_mean(tf.nn.relu(1 + fake_real_diff))
            scaled_loss = self.crt_opt.get_scaled_loss(loss)

        crt_vars = self.critic.trainable_variables
        scaled_crt_grads = crt_tape.gradient(scaled_loss, crt_vars)
        crt_grads = self.crt_opt.get_unscaled_gradients(scaled_crt_grads)
        self.crt_opt.apply_gradients(zip(crt_grads, crt_vars))
        return loss

    def build_generator(self):
        # Keep track of multi-scale generator outputs to feed to critic
        outputs = []
        input_layer = gan_layers.input_layer(shape=(gp.latent_dim,))

        # Input block
        gen = gan_layers.reshape(input_layer, shape=(1, 1, gp.latent_dim))
        gen = gan_layers.conv2d_transpose(gen, util.nf(0), kernel=4, padding='latent', normalize=None)
        gen = gan_layers.leaky_relu(gen)

        gen = gan_layers.conv2d(gen, util.nf(0), kernel=3)
        gen = gan_layers.leaky_relu(gen)
        outputs.append(gan_layers.ms_output_layer(gen, normalize=None))

        # Add the hidden generator blocks
        for block_res in range(self.target_res - 1):
            gen = gan_layers.nearest_neighbor(gen)
            gen = gan_layers.conv2d(gen, util.nf(block_res + 1), kernel=3)
            gen = gan_layers.leaky_relu(gen)

            gen = gan_layers.conv2d(gen, util.nf(block_res + 1), kernel=3)
            gen = gan_layers.leaky_relu(gen)
            outputs.append(gan_layers.ms_output_layer(gen, normalize=None))

        # Return finalized model TODO: Compile
        outputs = list(reversed(outputs))  # so that generator outputs and critic inputs are aligned
        return Model(inputs=input_layer, outputs=outputs)

    # TODO: This can be cleaned up a bit
    def build_critic(self):
        inputs = []
        exp_res = util.log_to_res(self.target_res)

        # Input layer (no concatenate in input layer)
        inputs.append(gan_layers.input_layer(shape=(exp_res, exp_res, 3)))
        crt = gan_layers.conv2d(inputs[-1], util.nf(self.target_res - 1), kernel=1, normalize=None)
        if gp.mbstd_in_each_layer:
            crt = gan_layers.minibatch_std(crt)

        crt = gan_layers.conv2d(crt, util.nf(self.target_res - 1), kernel=3, normalize=None)
        crt = gan_layers.leaky_relu(crt)

        crt = gan_layers.conv2d(crt, util.nf(self.target_res - 2), kernel=3, normalize=None)
        crt = gan_layers.leaky_relu(crt)
        crt = gan_layers.avg_pool(crt)

        # Intermediate layers
        for res in range(self.target_res - 1, 1, -1):
            exp_res = util.log_to_res(res)
            inputs.append(gan_layers.input_layer(shape=(exp_res, exp_res, 3)))

            # Multi-scale critic input
            crt = gan_layers.ms_input_layer(crt, inputs[-1], features=util.nf(res - 1), normalize=None)
            if gp.mbstd_in_each_layer:
                crt = gan_layers.minibatch_std(crt)
            crt = gan_layers.conv2d(crt, util.nf(res - 1), kernel=3, normalize=None)
            crt = gan_layers.leaky_relu(crt)

            crt = gan_layers.conv2d(crt, util.nf(res - 2), kernel=3, normalize=None)
            crt = gan_layers.leaky_relu(crt)
            crt = gan_layers.avg_pool(crt)

        # Output layer
        inputs.append(gan_layers.input_layer(shape=(4, 4, 3)))
        crt = gan_layers.ms_input_layer(crt, inputs[-1], features=util.nf(0), normalize=None)
        crt = gan_layers.minibatch_std(crt)

        crt = gan_layers.conv2d(crt, util.nf(0), kernel=3, normalize=None)
        crt = gan_layers.leaky_relu(crt)

        crt = gan_layers.conv2d(crt, util.nf(0), kernel=4, normalize=None)
        crt = gan_layers.leaky_relu(crt)

        crt = gan_layers.flatten(crt)
        crt = gan_layers.dense(crt, 1, dtype='float32')

        # Finalized model
        return Model(inputs=inputs, outputs=crt)


test_res = 64
msg = MsgGan(target_res=test_res)
util.pm(msg.generator, 'g' + str(msg.generator.output_shape[0][1]))
util.pm(msg.critic, 'c' + str(msg.critic.input_shape[0][1]))
msg.train()
