import tensorflow as tf
import datetime

# ================================== Training parameters ==================================
# Resolution for generated images (must be a power of 2)
target_res = 512

# Adam parameters
learning_rate = 0.001
beta_1 = 0.0
beta_2 = 0.99

# Training parameters
images_per_epoch = 600000
batch_size = 4

# Use mixed precision performance. Your GPU must have compute capability >7.0 to benefit from this, beyond
# some small memory and bandwidth gains
use_mixed_precision = True

# ================================== Layer parameters ==================================
lrelu_alpha = 0.2
normalize_latents = True

# Beware: my current implementation of convolutional concat uses 1.5x more memory and is 2x slower
concat_method = 'simple'  # 'conv' or 'simple'
mbstd_in_each_layer = False

# Use equalized learning rate. If you disable this lower the learning rate by at least an order of magnitude or
# else training diverges
use_elr = True

# ================================== Network parameters ==================================
# I think doing the <features> // 2 produces similar results for 64x64, but is *significantly* faster (rougly 2.5x)
max_features = 512
feature_base = 4096

latent_dim = 512

# Weight initialization standard deviation
init_stddev = 0.02

# Use 1e-7 instead of 1e-8 to avoid issues with mixed-precision underflow
pn_epsilon = 1e-7
mbstd_epsilon = 1e-7

# Number of batches per progress update
prog_update_freq = 5

# Number of progress updates per random image shown
updates_per_img = 10

# ================================== File/directory paths ==================================
run_id = datetime.datetime.now().strftime("%Y%m%d-[%H-%M-%S]")
model_dir = 'models/plots/' + run_id + '/'
model_weight_dir = 'models/weights/'
tensorboard_dir = 'logs/' + run_id + '/'
sample_output_dir = 'epoch_images/' + run_id + '/'
celeba_dir = 'celeba/'

# Maximum number of threads to use while pre-processing images
parallel_calls = tf.data.experimental.AUTOTUNE
prefetch_buffer = tf.data.experimental.AUTOTUNE

# number of images to keep in memory during training.
shuffle_buffer = 784

# If True, run tf.function-decorated functions in eager mode. Can be useful for debugging but usually breaks things
run_functions_eagerly = False

# If True, TensorFlow will allocate GPU memory as it needs it instead of all the GPU memory at once
gpu_grow_memory = True

# ================================== Other ==================================
figure_size = (19.2, 10.8)  # dpi=100
