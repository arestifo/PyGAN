import gan_params as gp
import tensorflow as tf
import glob
import pathlib
import datetime


def nf(res):
    return int(min(gp.feature_base / (2 ** res), gp.max_features))


def log_to_res(log):
    return int(2 ** (log + 1))  # +1 initial log is / 2. log2(2) = 1


def parse_image(image_path):
    image_string = tf.io.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    # Scale the image to [-1, 1]
    image = (image - 127.5) / 127.5
    return image


# returns a dataset of all image resolutions up to log_to_res(res)
def load_celeba(res):
    datasets = []
    resolutions = [log_to_res(r + 1) for r in range(res)]
    print(resolutions)
    for r in resolutions:
        print('Load ' + gp.celeba_dir + 'data%dx%d/' % (r, r))
        images = glob.glob(gp.celeba_dir + 'data%dx%d/' % (r, r) + '*.jpg')
        datasets.append(
            tf.data.Dataset.from_tensor_slices(images).map(parse_image, num_parallel_calls=gp.parallel_calls)
        )
    dataset = tf.data.Dataset.zip(tuple(reversed(datasets)))
    dataset = dataset.shuffle(gp.shuffle_buffer).repeat()

    dataset = dataset.batch(gp.batch_size)
    dataset = dataset.prefetch(gp.prefetch_buffer)
    return dataset.as_numpy_iterator()  # so I can use .next()


# make directory if it doesn't exist
def init_directory(dir_):
    pathlib.Path(dir_).mkdir(parents=True, exist_ok=True)


# generate a batch of latent vectors
def generate_latents():
    latents = tf.random.normal(shape=(gp.batch_size, gp.latent_dim))
    if gp.normalize_latents:
        latents = latents / tf.norm(latents, axis=-1, keepdims=True) * (gp.latent_dim ** 0.5)
    return latents


# plot model
def pm(model, fn):
    model_f = gp.model_dir
    pathlib.Path(model_f).mkdir(parents=True, exist_ok=True)
    tf.keras.utils.plot_model(model, to_file=model_f + fn + '.png', expand_nested=True, show_shapes=True)


def time_to_update(seen):
    return seen % (gp.batch_size * gp.prog_update_freq) == 0


def time_for_img(seen):
    return seen % (gp.batch_size * gp.prog_update_freq * gp.updates_per_img) == 0


# print list
def pl(obj):
    [print(_) for _ in obj]
