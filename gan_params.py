import tensorflow as tf
import datetime

# ================================== Training parameters ==================================
# Adam parameters
learning_rate = 0.0001
beta_1 = 0.0
beta_2 = 0.99
eps = 1e-8

# Training parameters
images_per_epoch = 600000
batch_size = 16

# Use mixed precision performance. Only Volta and later GPUs will benefit beyond small memory savings
use_mixed_precision = True

# ================================== Layer parameters ==================================
lrelu_alpha = 0.2
normalize_latents = True  # TODO: Get this working with spectral norm

# Beware: my current implementation of convolutional concat uses 1.5x more memory and is 2x slower
concat_method = 'simple'  # 'conv' or 'simple'

# Default normalization method. Can be overridden with normalize kwarg
default_norm = 'spectral'
mbstd_in_each_layer = True

# ================================== Network parameters ==================================
max_features = 512
feature_base = 4096

latent_dim = 512

# Weight initialization standard deviation
init_stddev = 0.02
pn_epsilon = 1e-8
mbstd_epsilon = 1e-8

# Number of batches per progress update
prog_update_freq = 5

# Number of progress updates per random image shown
updates_per_img = 15

# ================================== File/directory paths ==================================
run_id = datetime.datetime.now().strftime("%Y%m%d-[%H-%M-%S]")
model_dir = 'plots/' + run_id + '/'
tensorboard_dir = 'logs/' + run_id + '/'
sample_output_dir = 'epoch_images/' + run_id + '/'
celeba_dir = 'celeba/'

# Maximum number of threads to use while pre-processing images
parallel_calls = tf.data.experimental.AUTOTUNE
prefetch_buffer = tf.data.experimental.AUTOTUNE

# number of images to keep in memory during training.
shuffle_buffer = 784
