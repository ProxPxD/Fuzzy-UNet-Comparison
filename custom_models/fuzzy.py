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
        logging.debug(f'INIT: {self.input_dim=}, {self.output_dim=}')
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
        logging.debug(f'BUILT {init_centers.shape = }')
        logging.debug(f'BUILT {init_scales.shape = }')
        if init_centers.shape != init_scales.shape:
            raise Exception('initial_centers and initial_scales shapes does not match')

        diags = [np.insert(np.diag(s), self.input_dim, c, axis=1) for s, c in zip(init_scales, init_centers)]
        a = tf.convert_to_tensor(np.array(diags), dtype=tf.float32)
        return a

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
        logging.debug('')
        column_tensor = tf.one_hot(self.input_dim, self.input_dim+1)
        logging.debug(f'ConvFuzzyLayer: create_semi_widths: {column_tensor.shape = }')
        ones_tensor = tf.convert_to_tensor(np.array([column_tensor]*self.output_dim), dtype=tf.float32)
        logging.debug(f'ConvFuzzyLayer: create_semi_widths: {ones_tensor.shape = }')
        semi_widths = tf.reshape(ones_tensor, (self.output_dim, 1, self.input_dim+1))
        logging.debug(f'ConvFuzzyLayer: create_semi_widths: {semi_widths.shape = }')
        logging.debug('')
        return tf.Variable(semi_widths)

    def build(self, input_shape):
        logging.debug('BUILDING!')
        if input_changed := not self.input_dim:
            self.input_dim = input_shape[-1]
        logging.debug(f'BUILT {self.input_dim = }')
        if output_changed := not self.output_dim:
            self.output_dim = self.input_dim
        logging.debug(f'BUILT {self.output_dim = }')
        logging.debug(f'BUILT {input_changed = }, {output_changed = }')

        if not hasattr(self, 'A'):
            init_centers = tf.random.normal((self.output_dim, self.input_dim))
            init_scales = tf.ones((self.output_dim, self.input_dim))
            self.A = self.create_transformation_matrix(init_centers, init_scales)
        logging.debug(f'BUILT {self.A.shape = }')
        # if input_changed or output_changed or not hasattr(self, 'c_r'):  # Equivalent to hasattr(self, 'c_r')
        #     self.c_r = self.create_semi_widths()
        # logging.debug(f'BUILT {self.c_r.shape = }')

        logging.debug('')

    def call(self, X):
        logging.debug(f'{X.shape = }')
        X_with_ones = self.insert_ones(X)
        logging.debug(f'{X_with_ones.shape = }')
        tx = tf.transpose(X_with_ones, perm=[1, 0])
        logging.debug(f'{tx.shape = }')
        mul = self.A @ tx
        logging.debug(f'{mul.shape = }')
        logging.debug(f'{mul[:, :self.input_dim].shape = }')
        exponents = tf.norm(mul, axis=1)
        logging.debug(f'{exponents.shape = }')
        memberships = tf.exp(-exponents)
        logging.debug(f'{memberships.shape = }')
        transposed = tf.transpose(tf.expand_dims(memberships, axis=0))
        logging.debug(f'{transposed.shape = }')
        # raise NotImplementedError
        return transposed

    def insert_ones(self, X):
        repeated_one = tf.expand_dims(tf.ones(X.shape[:-1]), axis=-1)  # Shape with batch without channels
        ext_x = tf.concat([X, repeated_one], axis=-1)
        return ext_x

# class FuzzifyLayer(Layer, FuzzyOperations):
#
#     def __init__(self, *,
#             input_dims: Sequence[int, ...] = None,
#             output_dim: Shape = None,
#             initial_centers = None,
#             initial_scaling_factors = None,
#             **kwargs
#         ):
#
#         self.initial_centers = initial_centers
#         self.initial_sigmas = initial_scaling_factors
#         self.initial_scaling_factors = initial_scaling_factors
#
#         self.output_dim = self.centers = self.scaling_factors = None
#         self.input_dims = None
#
#         if input_dims:
#             self.build(utils.to_list(input_dims))
#
#         super(FuzzifyLayer, self).__init__(**kwargs)
#
#     @classmethod
#     def _create_input_dims(cls, input_shape: Shape):
#         return
#
#     def _create_centers(self, input_shape: Shape):
#         if self.initial_centers is None:
#             c_init_values = tf.random_uniform_initializer(-1, 1)(shape=(input_shape[-1], self.output_dim), dtype="float32")
#         else:
#             c_init_values = tf.convert_to_tensor(self.initial_centers, dtype="float32")
#         return tf.Variable(initial_value=c_init_values, trainable=True)
#
#     def _create_scaling_factors(self, input_shape: Shape):
#         if self.initial_scaling_factors is None:
#             a_init_values = tf.ones_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32")
#         else:
#             a_init_values = tf.convert_to_tensor(self.initial_scaling_factors, dtype="float32")
#         return tf.Variable(initial_value=a_init_values, trainable=True)
#
#     def build(self, input_shape):
#         self.input_dims = self._create_input_dims(input_shape)
#         self.output_dim = self.output_dim or self.input_dims[-1]
#         self.centers = self._create_centers(input_shape)
#         self.scaling_factors = self._create_scaling_factors(input_shape)
#         super(FuzzifyLayer, self).build(input_shape)
#
#     def call(self, x):
#         x = self.expand(x, self.output_dim, axis=-1)
#         c = self.expand_through(self.centers, self.input_dims)
#         a = self.expand_through(self.scaling_factors, self.input_dims)
#         xc = self.calc_membership(x, c, a)
#         return xc
#
#     @classmethod
#     def expand(cls, val, dim, *, axis=0):
#         return K.repeat_elements(K.expand_dims(val, axis), dim, axis)
#
#     @classmethod
#     def expand_through(cls, val, dims):
#         for dim in dims:
#             val = cls.expand(val, dim)
#         return val
#
#     @classmethod
#     def calc_membership(cls, x, c, a):
#         return K.exp(-K.sum(K.square((x - c) / (2 * a)), axis=-2, keepdims=False))
#
#     # def compute_output_shape(self, input_shape):
#     #     return tuple(input_shape[:-1]) + (self.output_dim,)
#
#
# class DefuzzifyLayer(Layer, FuzzyOperations):
#
#     def __init__(self,
#             output_dim: Shape = None,
#             initial_rules_outcomes=None,
#             input_shape: tuple[int, ...] = None,
#             input_dim: int = None,
#             **kwargs
#         ):
#         if not input_shape and input_dim:
#             input_shape = (input_dim,)
#
#         self.initial_rules_outcomes = initial_rules_outcomes
#         if input_shape:
#             self.input_dim = list(input_shape)[-1]
#             self.output_dim = output_dim if output_dim else self.input_dim
#             self.rules_outcome = self._create_rules_outcome
#         super(DefuzzifyLayer, self).__init__(**kwargs)
#
#     def _create_rules_outcome(self, input_shape):
#         if self.initial_rules_outcomes is None:
#             outcomes_init_values = tf.random_uniform_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32")
#         else:
#             outcomes_init_values = tf.convert_to_tensor(self.initial_rules_outcomes, dtype="float32")
#         return tf.Variable(initial_value=outcomes_init_values, trainable=True)
#
#     def build(self, input_shape):
#         self.input_dim = list(input_shape)[:-1:-1]
#         self.output_dim = self.output_dim if self.output_dim else self.input_dim
#         self.rules_outcome = self._create_rules_outcome
#         super(DefuzzifyLayer, self).build(input_shape)
#
#     def __call__(self, x):
#         x = self.expand(x, self.output_dim, axis=-1)
#         rules_outcome = self.expand_through(self.rules_outcome, self.input_dim)
#         xc = self.defuzify(x, rules_outcome)
#         return xc
#
#     def defuzzify(self, x, rules_outcome):
#         return K.sum((x * rules_outcome), axis=-2, keepdims=False)
#
#     # def compute_output_shape(self, input_shape):
#     #     return tuple(input_shape[:-1]) + (self.output_dim,)
#     #
#     # def get_config(self):
#     #     return {"rules_outcome": self.rules_outcome.numpy()}
#
#


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
            logging.debug(f'build self.{name} = {val}')

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
        # Shape: (#todo)
        logging.info(f'original x.shape = {x.shape}')

        batch_size = self.batch_size or x.shape[0] or 1
        has_channels = len(x.shape) - int(bool(x.shape[0])) > 2
        self._set_dims_from_shape(x.shape)

        channels = self.channels or (x.shape[-1] if has_channels else 1)
        logging.debug(f'{batch_size = }')
        logging.debug(f'{channels = }')
        logging.debug(f'\n')

        # Since the method does not differentiate between two channels of a single image any more that is
        # it differentiates between two channels of two separate images, it is beneficial to fold the first
        # two dimensions, i.e., batch_size and channels together.

        # x = tf.reshape(x, [batch_size * self.channels, *self.dims], name=None)
        # logging.debug(f'batch and channel merged x.shape = {x.shape}')

        # test_x = tf.reshape(x, [-1, *self.output_size, 1])
        # logging.debug(f'test: {test_x.shape}')
        # logging.debug(f'test: {self.extract_patches(test_x)}\n')
        logging.debug(f'{x.shape = }')
        logging.debug(f'{self.cols_padding, self.rows_padding = }')
        x = self.pad(x)
        self._set_dims_from_shape(x.shape)
        logging.debug(f'{self.output_size = }')
        logging.debug(f'padded {x.shape = }\n')
        # v_x = self.extract_patches(x, padding="VALID").shape
        # s_x = self.extract_patches(x, padding="SAME").shape
        # logging.debug(f'VALID: {v_x}, {[(d or 0) % self.kernel[0] for d in v_x]}')
        # logging.debug(f'SAME: {s_x}, {[(d or 0) % self.kernel[0] for d in s_x]}')

        x = self.extract_patches(x)
        logging.debug(f'patched {x.shape = }')

        logging.debug(f'{[-1, *self.output_size, self.n_tiles] = }')
        x = tf.reshape(x, [-1, *self.output_size, self.n_tiles])
        logging.debug(f'reshaped patched {x.shape = }\n')

        kmm = self.membership(x)
        v_avg = self.reduce_mean(kmm)
        var = self.calculate_variance(x, v_avg)

        logging.debug(f'{kmm.shape = }')
        logging.debug(f'{v_avg.shape = }')
        logging.debug(f'{var.shape = }')
        logging.debug('')
        pi = self.calculate_gaussian_membership(x, kmm, var)
        avg_pi = self.calculate_avg_pi(pi)
        thresh = self.calculate_threshold(pi)

        logging.debug('')
        logging.debug(f'{batch_size = }')
        logging.debug(f'{channels = }')

        # Conditions
        logging.debug(f'[helper] {kmm.shape = }')
        logging.debug(f'[helper] {var.shape = }')
        logging.debug(f'[helper] {avg_pi.shape = }')
        logging.debug(f'[helper] {thresh.shape = }\n')

        m_membership_importance = tf.reduce_any(avg_pi > thresh, axis=-1, keepdims=True)  # avg_pi[..., self.h-1] # new axis # mask_primary
        logging.debug(f'{m_membership_importance.shape = }')
        m_variance_importance = (var[..., self.h-1] < self.eps)[..., tf.newaxis]  # new axis  # s_condition
        logging.debug(f'{m_variance_importance.shape = }')

        # logging.debug(f'{(~m_membership_importance & m_variance_importance).shape = }')
        m_only_variance_importance = ~m_membership_importance & m_variance_importance
        logging.debug(f'{m_only_variance_importance.shape = }')
        # mask_secondary = tf.reduce_sum(tf.cast(~m_membership_importance & m_variance_importance, tf.float32), axis=-1, keepdims=True)
        # mask_secondary = tf.cast(mask_secondary > 0, tf.bool)  #
        m_unimportant = ~(m_membership_importance | m_only_variance_importance)
        logging.debug(f'{m_unimportant.shape = }')

        # joining channels and batch_size for simplicity
        pooled = tf.zeros([batch_size * channels * (self.n_tiles and 1), *self.output_size, 1])
        logging.debug(f'{pooled.shape = }')
        logging.debug(f'{x.shape = }')
        logging.debug('')

        # pooled operations
        pooled = tf.where(m_membership_importance, self.reduce_mean(x), pooled)  # self.reduce_mean(x)
        logging.debug(f'm_membership_importance {pooled.shape = }')
        pooled = tf.where(m_only_variance_importance, v_avg, pooled)
        logging.debug(f'm_only_variance_importance {pooled.shape = }')
        pooled = self.reduce_mean(pooled)
        logging.debug(f'reduced {pooled.shape = }')

        # count = int(tf.reduce_sum(tf.cast(m_unimportant, tf.float32)))  #.numpy()  # Count
        count = tf.reduce_sum(tf.cast(m_unimportant, tf.float32))#.numpy()  # Count
        count = count.numpy() if tf.executing_eagerly() else 0  # tf.reduce_any(count != 0)
        logging.debug(f'{count = }')

        if count:
            tiled_m_unimportant = self.tile(m_unimportant, shape=(count, *self.output_size, self.n_tiles))
            logging.debug(f'{tiled_m_unimportant.shape = }')

            region = tf.boolean_mask(x, tiled_m_unimportant)  # Region
            g = tf.boolean_mask(avg_pi, tiled_m_unimportant)  # g

            logging.debug(f'{region.shape = }')
            logging.debug(f'{g.shape = }')

            denoised = tf.reduce_sum(g*region, axis=-1, keepdims=True) / tf.reduce_sum(g, axis=-1, keepdims=True)
            denoised = tf.reshape(denoised, [count])  # Denoised
            logging.debug(f'{denoised.shape = }')

            indices = tf.where(m_unimportant)
            logging.debug(f'{indices.shape = }')
            pooled = tf.tensor_scatter_nd_update(pooled, indices, denoised)

            logging.debug(f'scattered {pooled.shape = }')

        logging.debug(f'pre reshaped {pooled.shape = }')
        pooled = tf.reshape(pooled, [batch_size, *self.output_size, channels])
        pooled = tf.identity(pooled)
        logging.info(f'final {pooled.shape = }')
        logging.debug('\n'*2)
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
        # logging.debug(f'Whole kmm mean is {kmm.shape}')
        for k in range(h-1):
            start = h-k-1
            end = h+k+1 if self.n_tiles % 2 == 0 else h + k
            x_partial_mean = self.reduce_mean(x, start, end)
            # logging.debug(f'partial kmm mean for {h=}, {k=}, x[..., {start}:{end}] is {x_partial_mean.shape}')
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
        logging.debug(f'{xrep.shape = }')
        logging.debug(f'{kmmrep.shape = }')
        logging.debug(f'{varrep.shape = }')
        logging.debug('')
        pi = tf.exp(-0.5 * tf.square((xrep - kmmrep)) / varrep)
        logging.debug(f'{pi.shape = }')
        return pi

    def calculate_threshold(self, pi: tf.Tensor) -> tf.Tensor:
        max_values = tf.reduce_max(pi, axis=3)
        logging.debug(f'{max_values.shape = }')
        thresh = tf.reduce_min(max_values, axis=3, keepdims=True)
        logging.debug(f'{thresh.shape = }')
        return thresh

    def calculate_avg_pi(self, pi: tf.Tensor) -> tf.Tensor:
        avg_pi = tf.reduce_mean(pi, axis=3)
        logging.debug(f'{avg_pi.shape = }')
        return avg_pi

    # kmmrep = tf.transpose(tf.reshape(tf.tile(kmm, [1, 1, 1, n]), [-1, *self.output_size, n, h]), perm=[0, 1, 2, 4, 3])
    # variancerep = tf.transpose(tf.reshape(tf.tile(variance, [1, 1, 1, n]), [-1, *self.output_size, n, h]), perm=[0, 1, 2, 4, 3])

    def tile(self, x: tf.Tensor, n: int = None, shape: Shape = None) -> tf.Tensor:
        n = n or x.shape[-1]
        shape = shape or [-1, *self.output_size, self.h, self.n_tiles]

        tiled = tf.tile(x, [1, 1, 1, n])
        reshaped = tf.reshape(tiled, shape)
        return reshaped

# class FuzzyLayer(Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.fuzzify = FuzzifyLayer()
#         self.defuzzify = DefuzzifyLayer()
#
#     def __call__(self, x):
#         x = self.fuzzify(x)
#         x = self.defuzzify(x)
#         return x
