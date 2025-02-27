import logging
import operator as op
from functools import reduce

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logging.debug('')

Shape = tuple[int, ...]
Shapable = Shape | int


class FuzzyOperations:
    @classmethod
    def expand(cls, val, dim, *, axis=0):
        return K.repeat_elements(K.expand_dims(val, axis), dim, axis)

    @classmethod
    def expand_through(cls, val, dims):
        for dim in dims:
            val = cls.expand(val, dim)
        return val


class FuzzyLayer(Layer):
    def __init__(self, *,
            output_dim: int = None,
            input_dim: int = None,
            init_centers = None,
            init_scales = None,
    ):
        """
        mu_j(x,a,c) = exp(-|| a . x ||^2)
        """
        super().__init__()  # TODO: move to build?
        self.output_dim = output_dim
        self.input_dim = input_dim
        if init_centers and init_scales:
            self.A = self.create_transformation_matrix(init_centers, init_scales)
        self.ta = None
        if self.input_dim and self.output_dim:
            self.c_r = self.create_semi_widths()

    def create_transformation_matrix(self, init_scales, init_centers):
        """
        :param init_scales: A scale (diagonal values s), representing the width or spread of the fuzzy number.
        :param init_centers: A center (the inserted column c), representing the mean or central value of the fuzzy number.
        :return: Each slice looks like:
            [[s1, 0,  0,  c1],
             [0,  s2, 0,  c2],
             [0,  0,  s3, c3]]
        """
        if init_centers.shape != init_scales.shape:
            raise Exception('initial_centers and initial_scales shapes does not match')

        diags = [np.insert(np.diag(s), self.input_dim, c, axis=1) for s, c in zip(init_scales, init_centers)]
        a = tf.convert_to_tensor(np.array(diags), dtype=tf.float32)
        return tf.Variable(a)

    def create_semi_widths(self):
        """
        Creates the initial semi-widths matrix for fuzzy logic membership functions.

        This function generates a tensor that represents the initial spread (semi-width) of fuzzy sets
        based on input dimensions and a predefined output dimension. It creates a one-hot encoded tensor
        where the last feature (center) has a value of 1, and all others have 0. This tensor is used
        to initialize the semi-width values across the different output fuzzy sets.

        :return:
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]]

        """
        column_tensor = tf.one_hot(self.input_dim, self.input_dim+1)
        ones_tensor = tf.convert_to_tensor(np.array([column_tensor]*self.output_dim), dtype=tf.float32)
        semi_widths = tf.reshape(ones_tensor, (self.output_dim, 1, self.input_dim+1))
        return tf.Variable(semi_widths)

    def build(self, input_shape):
        if input_changed := not self.input_dim:
            self.input_dim = input_shape[-1]
        if output_changed := not self.output_dim:
            self.output_dim = self.input_dim

        if not hasattr(self, 'A'):
            init_centers = tf.random.normal((self.output_dim, self.input_dim))
            init_scales = tf.ones((self.output_dim, self.input_dim))
            self.A = self.create_transformation_matrix(init_centers, init_scales)
        if input_changed or output_changed or not hasattr(self, 'c_r'):  # Equivalent to hasattr(self, 'c_r')
            self.c_r = self.create_semi_widths()

        self.ta = self.concatenate_a_and_cr()


    def concatenate_a_and_cr(self):
        """
        :return: Each slice looks like:
            [[s1, 0,  0,  c1],
             [0,  s2, 0,  c2],
             [0,  0,  s3, c3]
             [0,  0,  0,  1]]
        """
        return tf.concat([self.A, self.c_r], axis=1)

    def call(self, X):
        X_with_ones = self.insert_ones(X)
        tx = tf.transpose(X_with_ones, perm=[1, 0])
        mul = self.ta @ tx
        exponents = tf.norm(mul[:, :self.input_dim], axis=1)
        memberships = tf.exp(-exponents)
        transposed = tf.transpose(tf.expand_dims(memberships, axis=0))
        return transposed

    def insert_ones(self, X):
        repeated_one = tf.expand_dims(tf.ones(X.shape[:-1]), axis=-1)  # Shape with batch without channels
        ext_x = tf.concat([X, repeated_one], axis=-1)
        return ext_x


class FuzzyPooling(Layer):
    """Custom layer for Type-2 Fuzzy logic based pooling"""
    def __init__(self,
            kernel: Shapable = 3,
            stride: Shapable = None,
            channels: int = None,
            dims: Shape = None,
            batch_size: int = None,
            eps: int = None,
        ):
        super().__init__()
        self.batch_size: int = batch_size
        self.dims: Shape = dims
        self.channels: int = channels

        self.stride: Shape = stride if isinstance(stride, tuple) else (stride := stride or kernel, stride)
        self.kernel: Shape = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        # half the Kernel size squared, rounded up
        # it represents a fuzzy set that covers roughly half of the kernel elements
        # or the number of fuzzy numbers set in the middle varying by one
        self.h: int = (self.n_tiles + 1) // 2
        self.eps = eps or 1e-4

    @property
    def rows(self) -> int:
        return self.dims[0]

    @property
    def cols(self) -> int:
        return self.dims[1]

    """Number of kernel tiles (kernel_rows * kernel_cols)"""
    @property
    def n_tiles(self) -> int:
        return reduce(op.mul, self.kernel)

    @property
    def kernel_rows(self) -> int:
        return self.kernel[0]

    @property
    def kernel_cols(self) -> int:
        return self.kernel[1]

    @property
    def stride_by_rows(self) -> int:
        return self.stride[0]

    @property
    def stride_by_cols(self) -> int:
        return self.stride[1]

    def _calculate_output_size(self, n: int) -> int:  # Adjust for padding
        return (self.dims[n] - self.kernel[n]) // self.stride[n] + 1

    @property
    def output_rows(self) -> int:
        return self._calculate_output_size(0)

    @property
    def output_cols(self) -> int:
        return self._calculate_output_size(1)

    @property
    def output_size(self) -> tuple[int, int]:
        return self.output_rows, self.output_cols

    @property
    def rows_padding(self) -> int:
        return (self.kernel_rows - self.rows % self.kernel_rows) % self.kernel_rows

    @property
    def cols_padding(self) -> int:
        return (self.kernel_cols - self.cols % self.kernel_cols) % self.kernel_cols

    def build(self, input_shape):
        self._set_from_shape('batch_size', input_shape, 0, 0)
        self._set_from_shape('channels', input_shape, -1)
        self._set_from_shape('dims', input_shape, slice(1, 3))

    @tf.autograph.experimental.do_not_convert
    def _set_dims_from_shape(self, shape):
        self._set_from_shape('dims', shape, slice(1, 3))

    @tf.autograph.experimental.do_not_convert
    def _set_from_shape(self, name: str, shape, s: slice | int, if_n: int = None) -> None:
        if if_n is None or shape[if_n]:
            setattr(self, name, val := shape[s])

    def calculate_fuzzy_variance(self, x, v_avg, eps=1e-4):
        h = self.h
        omega = abs(x - v_avg)
        sigma = self.reduce_mean(omega)

        for k in range(h-1):
            start = h-k-1
            end = h+k+1 if self.n_tiles % 2 == 0 else h + k
            partial_mean = self.reduce_mean(omega, start, end)
            sigma = tf.concat([partial_mean, sigma], axis=-1)
        return sigma + eps

    def call(self, x):
        batch_size = self.batch_size or x.shape[0] or 1
        has_channels = len(x.shape) - int(bool(x.shape[0])) > 2
        self._set_dims_from_shape(x.shape)

        channels = self.channels or (x.shape[-1] if has_channels else 1)

        # Since the method does not differentiate between two channels of a single image any more that is
        # it differentiates between two channels of two separate images, it is beneficial to fold the first
        # two dimensions, i.e., batch_size and channels together.

        x = self.pad(x)
        self._set_dims_from_shape(x.shape)
        x = self.extract_patches(x)
        x = tf.reshape(x, [-1, *self.output_size, self.n_tiles])

        kmm = self.membership(x)
        v_avg = self.reduce_mean(kmm)
        var = self.calculate_variance(x, v_avg)

        pi = self.calculate_gaussian_membership(x, kmm, var)
        avg_pi = self.calculate_avg_pi(pi)
        thresh = self.calculate_threshold(pi)

        m_membership_importance = tf.reduce_any(avg_pi > thresh, axis=-1, keepdims=True)  # avg_pi[..., self.h-1] # new axis # mask_primary
        m_variance_importance = (var[..., self.h-1] < self.eps)[..., tf.newaxis]  # new axis  # s_condition

        m_only_variance_importance = ~m_membership_importance & m_variance_importance
        m_unimportant = ~(m_membership_importance | m_only_variance_importance)

        # joining channels and batch_size for simplicity
        pooled = tf.zeros([batch_size * channels * (self.n_tiles and 1), *self.output_size, 1])

        # pooled operations
        pooled = tf.where(m_membership_importance, self.reduce_mean(x), pooled)  # self.reduce_mean(x)
        pooled = tf.where(m_only_variance_importance, v_avg, pooled)
        pooled = self.reduce_mean(pooled)

        count = tf.reduce_sum(tf.cast(m_unimportant, tf.float32))#.numpy()  # Count
        count = count.numpy() if tf.executing_eagerly() else 0  # tf.reduce_any(count != 0)

        if count:
            tiled_m_unimportant = self.tile(m_unimportant, shape=(count, *self.output_size, self.n_tiles))

            region = tf.boolean_mask(x, tiled_m_unimportant)  # Region
            g = tf.boolean_mask(avg_pi, tiled_m_unimportant)  # g

            denoised = tf.reduce_sum(g*region, axis=-1, keepdims=True) / tf.reduce_sum(g, axis=-1, keepdims=True)
            denoised = tf.reshape(denoised, [count])  # Denoised

            indices = tf.where(m_unimportant)
            pooled = tf.tensor_scatter_nd_update(pooled, indices, denoised)

        pooled = tf.reshape(pooled, [batch_size, *self.output_size, channels])
        pooled = tf.identity(pooled)
        return pooled

    def pad(self, x):
        return tf.pad(x, [[0, 0], [0, self.rows_padding], [0, self.cols_padding], [0, 0]])

    def extract_patches(self,
            x,
            *,
            padding: str = None):
        # Shape: (#todo)
        # ..., channels * tiles
        # last dimension is arranged channel by channel
        return tf.image.extract_patches(x,
            sizes=[1, *self.kernel, 1],
            strides=[1, *self.stride, 1],
            rates=[1, 1, 1, 1],
            padding=padding or 'SAME'
        )

    def membership(self, x: tf.Tensor) -> tf.Tensor:
        h = self.h
        # Kernel Mean Matrix
        kmm = self.reduce_mean(x)
        for k in range(h-1):
            start = h-k-1
            end = h+k+1 if self.n_tiles % 2 == 0 else h + k
            x_partial_mean = self.reduce_mean(x, start, end)
            kmm = tf.concat([x_partial_mean, kmm], axis=-1)

        return kmm

    @classmethod
    def reduce_mean(cls, x: tf.Tensor, start=None, end=None):
        return tf.reduce_mean(x[..., start:end], axis=-1, keepdims=True)

    def calculate_variance(self, x: tf.Tensor, avg: tf.Tensor, eps=None) -> tf.Tensor:
        omega = abs(x - avg)
        variance = self.membership(omega)
        return variance + (eps or self.eps)

    def calculate_gaussian_membership(self, x: tf.Tensor, kmm: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
        xrep = self.tile(x, self.h)
        kmmrep = self.tile(kmm, self.n_tiles)
        varrep = self.tile(var, self.n_tiles)
        pi = tf.exp(-0.5 * tf.square((xrep - kmmrep)) / varrep)
        return pi

    def calculate_threshold(self, pi: tf.Tensor) -> tf.Tensor:
        max_values = tf.reduce_max(pi, axis=3)
        thresh = tf.reduce_min(max_values, axis=3, keepdims=True)
        return thresh

    def calculate_avg_pi(self, pi: tf.Tensor) -> tf.Tensor:
        avg_pi = tf.reduce_mean(pi, axis=3)
        return avg_pi

    def tile(self, x: tf.Tensor, n: int = None, shape: Shape = None) -> tf.Tensor:
        n = n or x.shape[-1]
        shape = shape or [-1, *self.output_size, self.h, self.n_tiles]

        tiled = tf.tile(x, [1, 1, 1, n])
        reshaped = tf.reshape(tiled, shape)
        return reshaped
