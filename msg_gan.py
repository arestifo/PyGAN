import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
import os
import itertools
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import gan_layers as gl
import gan_util as util
import gan_params as gp

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], gp.gpu_grow_memory)
tf.config.experimental_run_functions_eagerly(gp.run_functions_eagerly)
plt.rcParams["figure.figsize"] = gp.figure_size


class MsgGan:
    def __init__(self):
        assert np.ceil(np.log2(gp.target_res)) == \
               np.floor(np.log2(gp.target_res)), 'Target resolution must be a power of 2'

        self.target_res = int(np.log2(gp.target_res / 2))

        self.gen_opt = optimizers.Adam(learning_rate=gp.learning_rate, beta_1=gp.beta_1, beta_2=gp.beta_2)
        self.crt_opt = optimizers.Adam(learning_rate=gp.learning_rate, beta_1=gp.beta_1, beta_2=gp.beta_2)

        if gp.use_mixed_precision:
            self.gen_opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.gen_opt)
            self.crt_opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.crt_opt)

        # Build models
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Seed for keeping track of progress
        self.seed = util.generate_latents()

        # Set up directories
        util.init_directory(gp.tensorboard_dir)  # log dir
        util.init_directory(gp.sample_output_dir)  # generated sample dir
        util.init_directory(gp.model_dir)  # model architecture plot dir
        util.init_directory(gp.model_weight_dir)  # saved model weights dir

        # Set up TensorBoard logging
        self.log_writer = tf.summary.create_file_writer(gp.tensorboard_dir)
        self.total_seen = 0  # keep track of total number of images seen
        self.total_batches = 0  # keep track of total batches sen

        # Write model summaries to TensorBoard for debugging
        self.write_model_summaries()

        # Create CheckpointManager to save model state during training
        self.checkpoint_mgr = self.create_checkpoint_manager()

        # Progress bar format
        self.pb_widgets = [
            '[', pb.Variable('batches', format='{formatted_value}'), '] ', pb.Variable('g_loss', precision=10), ', ',
            pb.Variable('c_loss', precision=10), ' [', pb.SimpleProgress(), '] ',
            pb.FileTransferSpeed(unit='img', prefixes=['', 'k', 'm']), '\t', pb.ETA()
        ]

    def write_model_summaries(self):
        with self.log_writer.as_default():
            tf.summary.text('generator', self.generator.to_json(), step=0)
            tf.summary.text('critic', self.critic.to_json(), step=0)

    def create_checkpoint_manager(self):
        checkpoint = tf.train.Checkpoint(generator=self.generator, critic=self.critic,
                                         gen_opt=self.gen_opt, crt_opt=self.crt_opt,
                                         seed=self.seed, total_seen=self.total_seen, total_batches=self.total_batches)
        return tf.train.CheckpointManager(checkpoint, directory=gp.model_weight_dir, max_to_keep=3)

    def random_image(self, show=True):
        r_img = self.generator(util.generate_latents())
        self.view_imgs(r_img, show=show)

    # view multi-scale generator output
    def view_imgs(self, images, show=False, rows=4):
        # print(np.shape(images))
        assert rows <= gp.batch_size, 'Number of rows cannot exceed batch size'

        fig, axs = plt.subplots(nrows=rows, ncols=self.target_res)

        image_indices = itertools.product(range(self.target_res), range(rows))
        for row, col in image_indices:
            image = images[row][col]
            image = (image - np.min(image)) / np.ptp(image)  # Scale images to [0, 1]
            axs[col, row].axis('off')
            axs[col, row].imshow(image)

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(gp.sample_output_dir, str(self.total_seen) + '.png'))

        # Clean up to avoid memory leaks
        plt.cla()  # close axis
        plt.close(fig)

    # TODO: Implement exponential moving averages for the generator weights
    # TODO: Implement multi-GPU support
    def train(self):
        epochs = 10

        dataset = util.load_celeba(self.target_res)
        for epoch in range(epochs):
            epoch_seen = 0
            with pb.ProgressBar(widgets=self.pb_widgets, max_value=gp.images_per_epoch) as progress:
                while epoch_seen < gp.images_per_epoch:
                    # Get a new batch of real images and create new generator input
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

                    # TODO: Implement with callbacks
                    if util.time_to_update(epoch_seen):
                        progress.update(epoch_seen, batches=self.total_batches,
                                        g_loss=generator_loss, c_loss=critic_loss)

                    if util.time_for_img(epoch_seen):
                        sample_imgs = self.generator(self.seed)
                        self.view_imgs(sample_imgs)

                # Save model weights after each epoch
                self.checkpoint_mgr.save()

    # Hypothesis: getting the grads of the MEAN hinge loss loses batch data and destabilizes the training process
    # Relativistic Hinge loss (RaHinge)
    # See SAGAN paper: https://arxiv.org/pdf/1805.08318.pdf
    @tf.function
    def train_generator(self, real_batch, latent_input):
        with tf.GradientTape() as gen_tape:
            fake_batch = self.generator(latent_input)
            real_preds = self.critic(real_batch)
            fake_preds = self.critic(fake_batch)

            real_fake_diff = real_preds - tf.reduce_mean(fake_preds)
            fake_real_diff = fake_preds - tf.reduce_mean(real_preds)

            loss = tf.reduce_mean(tf.nn.relu(1 + real_fake_diff)) + tf.reduce_mean(tf.nn.relu(1 - fake_real_diff))
            scaled_loss = self.gen_opt.get_scaled_loss(loss)

        scaled_gen_grads = gen_tape.gradient(scaled_loss, self.generator.trainable_variables)
        gen_grads = self.gen_opt.get_unscaled_gradients(scaled_gen_grads)

        self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        return loss

    @tf.function
    def train_critic(self, real_batch, latent_input):
        with tf.GradientTape() as crt_tape:
            fake_batch = self.generator(latent_input)
            real_preds = self.critic(real_batch)
            fake_preds = self.critic(fake_batch)

            real_fake_diff = real_preds - tf.reduce_mean(fake_preds)
            fake_real_diff = fake_preds - tf.reduce_mean(real_preds)

            loss = tf.reduce_mean(tf.nn.relu(1 - real_fake_diff)) + tf.reduce_mean(tf.nn.relu(1 + fake_real_diff))
            scaled_loss = self.crt_opt.get_scaled_loss(loss)

        scaled_crt_grads = crt_tape.gradient(scaled_loss, self.critic.trainable_variables)
        crt_grads = self.crt_opt.get_unscaled_gradients(scaled_crt_grads)

        self.crt_opt.apply_gradients(zip(crt_grads, self.critic.trainable_variables))
        return loss

    # TODO: Implement fused upscale & downscaled
    def build_generator(self):
        # Keep track of multi-scale generator outputs to feed to critic
        outputs = []
        input_layer = gl.input_layer(shape=(gp.latent_dim,))

        # Input block
        gen = gl.dense(input_layer, 4 * 4 * util.nf(0))
        gen = gl.reshape(gen, shape=(4, 4, util.nf(0)))
        gen = gl.leaky_relu(gen)
        gen = gl.normalize(gen, method='pixel_norm')

        gen = gl.conv2d(gen, util.nf(0), kernel=3)
        gen = gl.leaky_relu(gen)
        gen = gl.normalize(gen, method='pixel_norm')
        outputs.append(gl.to_rgb(gen))

        # Add the hidden generator blocks
        for block_res in range(self.target_res - 1):
            gen = gl.nearest_neighbor(gen)
            gen = gl.conv2d(gen, util.nf(block_res + 1), kernel=3)
            gen = gl.leaky_relu(gen)
            gen = gl.normalize(gen, method='pixel_norm')

            gen = gl.conv2d(gen, util.nf(block_res + 1), kernel=3)
            gen = gl.leaky_relu(gen)
            gen = gl.normalize(gen, method='pixel_norm')
            outputs.append(gl.to_rgb(gen))

        # Return finalized model TODO: Compile
        outputs = list(reversed(outputs))  # so that generator outputs and critic inputs are aligned
        return Model(inputs=input_layer, outputs=outputs)

    # TODO: This can be cleaned up a bit
    def build_critic(self):
        inputs = []
        exp_res = util.log_to_res(self.target_res)

        # Input layer (no concatenate in input layer)
        inputs.append(gl.input_layer(shape=(exp_res, exp_res, 3)))
        crt = gl.conv2d(inputs[-1], util.nf(self.target_res - 1), kernel=1)
        if gp.mbstd_in_each_layer:
            crt = gl.minibatch_std(crt)

        crt = gl.conv2d(crt, util.nf(self.target_res - 1), kernel=3)
        crt = gl.leaky_relu(crt)

        crt = gl.conv2d(crt, util.nf(self.target_res - 2), kernel=3)
        crt = gl.leaky_relu(crt)
        crt = gl.avg_pool(crt)

        # Intermediate layers
        for res in range(self.target_res - 1, 1, -1):
            exp_res = util.log_to_res(res)
            inputs.append(gl.input_layer(shape=(exp_res, exp_res, 3)))

            # Multi-scale critic input
            crt = gl.combine(crt, inputs[-1], features=util.nf(res - 1))
            if gp.mbstd_in_each_layer:
                crt = gl.minibatch_std(crt)
            crt = gl.conv2d(crt, util.nf(res - 1), kernel=3)
            crt = gl.leaky_relu(crt)

            crt = gl.conv2d(crt, util.nf(res - 2), kernel=3)
            crt = gl.leaky_relu(crt)
            crt = gl.avg_pool(crt)

        # Output layer
        inputs.append(gl.input_layer(shape=(4, 4, 3)))
        crt = gl.combine(crt, inputs[-1], features=util.nf(0))
        crt = gl.minibatch_std(crt)

        crt = gl.conv2d(crt, util.nf(0), kernel=3)
        crt = gl.leaky_relu(crt)

        crt = gl.flatten(crt)
        crt = gl.dense(crt, util.nf(0))
        crt = gl.leaky_relu(crt)

        crt = gl.dense(crt, 1, dtype='float32')

        # Finalized model
        return Model(inputs=inputs, outputs=crt)


msg = MsgGan()
util.pm(msg.generator, 'g' + str(msg.generator.output_shape[0][1]))
util.pm(msg.critic, 'c' + str(msg.critic.input_shape[0][1]))
