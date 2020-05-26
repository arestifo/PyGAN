import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Dense, Input, LeakyReLU, Flatten, Reshape
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Concatenate, Add, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.initializers import RandomNormal, he_normal
import gan_params as gp

from SpectralNormalizationKeras import SpectralNorm


# based on code from https://tinyurl.com/yctmhav7 (Jason Brownlee, Machine Learning Mastery)
class MinibatchStd(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)

        b_mean = inputs - tf.reduce_mean(inputs, axis=0, keepdims=True)
        b_stddev = tf.sqrt(tf.reduce_mean(tf.square(b_mean), axis=0) + gp.mbstd_epsilon)
        avg_stddev = tf.reshape(tf.reduce_mean(b_stddev), shape=(1, 1, 1, 1))
        avg_std_fm = tf.tile(avg_stddev, (shape[0], shape[1], shape[2], 1))
        return tf.concat([inputs, avg_std_fm], axis=-1)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1  # batch-wide std adds one additional channel
        return tuple(input_shape)


# ProGAN paper: pixelwise feature vector normalization layer
class PixelNorm(Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PixelNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs *= tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + gp.pn_epsilon)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


def minibatch_std(in_layer):
    return MinibatchStd()(in_layer)


def conv2d(in_layer, features, kernel, stride=(1, 1), normalize=gp.default_norm,
           constraint=None, init=he_normal(), **kwargs):
    if 'padding' in kwargs:
        if kwargs.pop('padding') == 'latent':
            in_layer = ZeroPadding2D(padding=((0, 3), (3, 0)))(in_layer)
    conv = Conv2D(features, (kernel, kernel), strides=stride, padding='same',
                  kernel_constraint=constraint, kernel_initializer=init, **kwargs)
    return _normalize(conv, in_layer, normalize) if normalize else conv(in_layer)


def conv2d_transpose(in_layer, features, kernel, stride=(1, 1), normalize=gp.default_norm,
                     constraint=None, init=he_normal(), **kwargs):
    if 'padding' in kwargs:
        if kwargs.pop('padding') == 'latent':
            in_layer = ZeroPadding2D(padding=((0, 3), (3, 0)))(in_layer)
    conv_t = Conv2DTranspose(features, (kernel, kernel), strides=stride, padding='same',
                             kernel_constraint=constraint, kernel_initializer=init, **kwargs)
    return _normalize(conv_t, in_layer, normalize) if normalize else conv_t(in_layer)


def dense(in_layer, features, constraint=None, init=he_normal(), **kwargs):
    return Dense(features, kernel_constraint=constraint, kernel_initializer=init, **kwargs)(in_layer)


def leaky_relu(in_layer):
    return LeakyReLU(alpha=gp.lrelu_alpha)(in_layer)


def flatten(in_layer):
    return Flatten()(in_layer)


def input_layer(shape):
    return Input(shape=shape)


def _normalize(layer, in_layer, norm_method):
    if norm_method == 'pixel':  # pixel-wise feature vector normalization
        in_layer = PixelNorm()(layer(in_layer))
    elif norm_method == 'spectral':
        in_layer = SpectralNorm(layer)(in_layer)
    elif norm_method == 'bn':
        in_layer = BatchNormalization()(in_layer)
    elif norm_method == 'ln':
        in_layer = LayerNormalization()(in_layer)
    else:
        raise ValueError('Invalid normalization method')
    return in_layer


def avg_pool(in_layer):
    return AveragePooling2D()(in_layer)


def nearest_neighbor(in_layer):
    return UpSampling2D()(in_layer)


def reshape(in_layer, shape):
    return Reshape(target_shape=shape)(in_layer)


# intermediate to-RGB generator block
def ms_output_layer(in_layer, **kwargs):
    norm = None if 'normalize' not in kwargs else kwargs.pop('normalize')
    return conv2d(in_layer, 3, kernel=1, dtype='float32', normalize=norm, **kwargs)


# intermediate concat layer
# concats previous critic layer activations (R x R x N_Features(R+1)) with either:
# - a randomly generated image (R x R x 3)
# - a downsampled real image (R x R x 3)
def ms_input_layer(prev_layer, image_input, features=None, **kwargs):
    # channel-wise concatenation
    if gp.concat_method == 'simple':
        return Concatenate(axis=-1)([image_input, prev_layer])
    elif gp.concat_method == 'conv':
        print('hi')
        if features:
            norm = None if 'normalize' not in kwargs else kwargs.pop('normalize')
            image_input = conv2d(image_input, features, kernel=1, dtype='float32', normalize=norm, **kwargs)  # to-rgb layer
            return Concatenate(axis=-1)([image_input, prev_layer])
        raise ValueError('Number of features is `None` when using conv concat method')
    raise ValueError('Invalid concatenation method')
