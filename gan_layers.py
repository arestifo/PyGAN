import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Dense, Input, LeakyReLU, Flatten, Reshape
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Concatenate, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.initializers import RandomNormal, he_normal
import gan_params as gp

from SpectralNormalizationKeras import SpectralNorm


# based on code from https://tinyurl.com/yctmhav7 (Jason Brownlee, Machine Learning Mastery)
class MinibatchStd(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = tf.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = tf.reduce_mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += gp.mbstd_epsilon
        # square root of the variance (stdev)
        stdev = tf.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = tf.reduce_mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = tf.shape(inputs)
        output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = tf.concat([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
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


# Using SpectralNorm on ALL (inc 1x1, 3x3, 4x4) layers doesn't seem to work
def conv2d(in_layer, features, kernel, stride=(1, 1), constraint=None, init=he_normal(), **kwargs):
    if 'padding' in kwargs:
        if kwargs.pop('padding') == 'latent':
            in_layer = ZeroPadding2D(padding=((0, 3), (3, 0)))(in_layer)
    conv = Conv2D(features, (kernel, kernel), strides=stride, padding='same',
                  kernel_constraint=constraint, kernel_initializer=init, **kwargs)

    # MSG-GAN: Apply spectral normalization to all 3x3 convolutions
    if kernel == 3:
        conv = SpectralNorm(conv)
    return conv(in_layer)


def conv2d_transpose(in_layer, features, kernel, stride=(1, 1), constraint=None, init=he_normal(), **kwargs):
    if 'padding' in kwargs:
        if kwargs.pop('padding') == 'latent':
            # scale up from 1x1xlatent_dim to 4x4xlatent_dim by padding with zeros
            in_layer = reshape(in_layer, shape=(1, 1, gp.latent_dim))
            in_layer = ZeroPadding2D(padding=((0, 3), (3, 0)))(in_layer)
    conv_t = Conv2DTranspose(features, (kernel, kernel), strides=stride, padding='same',
                             kernel_constraint=constraint, kernel_initializer=init, **kwargs)
    # MSG-GAN: Apply spectral normalization to all 3x3 convolutions
    if kernel == 3:
        conv_t = SpectralNorm(conv_t)
    return conv_t(in_layer)


def dense(in_layer, features, spectral=False, constraint=None, init=he_normal(), **kwargs):
    dense_ = Dense(features, kernel_constraint=constraint, kernel_initializer=init, **kwargs)

    return SpectralNorm(dense_)(in_layer) if spectral else dense_(in_layer)


def leaky_relu(in_layer):
    return LeakyReLU(alpha=gp.lrelu_alpha)(in_layer)


def flatten(in_layer):
    return Flatten()(in_layer)


def input_layer(shape):
    return Input(shape=shape)


def normalize(in_layer, method):
    if method == 'pixel':
        return PixelNorm()(in_layer)
    elif method == 'bn':
        return BatchNormalization()(in_layer)
    elif method == 'ln':
        return LayerNormalization()(in_layer)
    raise ValueError('Invalid normalization method')


def avg_pool(in_layer):
    return AveragePooling2D()(in_layer)


def nearest_neighbor(in_layer):
    return UpSampling2D()(in_layer)


def reshape(in_layer, shape):
    return Reshape(target_shape=shape)(in_layer)


# intermediate to-RGB generator block
def ms_output_layer(in_layer, **kwargs):
    return conv2d(in_layer, 3, kernel=1, dtype='float32', **kwargs)


# intermediate concat layer
# concats previous critic layer activations (R x R x N_Features(R+1)) with either:
# - a randomly generated image (R x R x 3)
# - a downsampled real image (R x R x 3)
def ms_input_layer(prev_layer, image_input, features=None, **kwargs):
    # channel-wise concatenation
    if gp.concat_method == 'simple':
        return Concatenate(axis=-1)([image_input, prev_layer])
    elif gp.concat_method == 'conv':
        if features:
            image_input = conv2d(image_input, features, kernel=1, dtype='float32', **kwargs)  # to-rgb layer
            return Concatenate(axis=-1)([image_input, prev_layer])
        raise ValueError('Number of features is `None` when using conv concat method')
    raise ValueError('Invalid concatenation method')
