import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, concatenate
from toolz import compose, compose_left

Shape = tuple[int, ...]


class FuzzyOperations:
    @classmethod
    def expand(cls, val, dim, *, axis=0):
        return K.repeat_elements(K.expand_dims(val, axis), dim, axis)

    @classmethod
    def expand_through(cls, val, dims):
        for dim in dims:
            val = cls.expand(val, dim)
        return val


class FuzzifyLayer(Layer, FuzzyOperations):

    def __init__(self,
            output_dim: Shape = None,
            initial_centers = None,
            initial_scaling_factors=None,
            input_shape: tuple[int, ...] = None,
            input_dim: int = None,
            **kwargs
        ):
        if not input_shape and input_dim:
            input_shape = (input_dim,)
        self.initial_centers = initial_centers
        self.initial_sigmas = initial_scaling_factors
        if input_shape:
            self.input_dims = self._create_input_dimensions(input_shape)
            self.output_dims = output_dim if output_dim else self.input_dims
            self.centers = self._create_centers(input_shape)
            self.a = self._create_scaling_factors(input_shape)
        super(FuzzifyLayer, self).__init__(**kwargs)

    @classmethod
    def _create_input_dimensions(cls, input_shape: Shape):
        return list(input_shape)[:-1:-1]

    def _create_centers(self, input_shape: Shape):
        if self.initial_centers is None:
            c_init_values = tf.random_uniform_initializer(-1, 1)(shape=(input_shape[-1], self.output_dims), dtype="float32")
        else:
            c_init_values = tf.convert_to_tensor(self.initial_centers, dtype="float32")
        return tf.Variable(initial_value=c_init_values, trainable=True)

    def _create_scaling_factors(self, input_shape: Shape):
        if self.initial_scaling_factors is None:
            a_init_values = tf.ones_initializer()(shape=(input_shape[-1], self.output_dims), dtype="float32")
        else:
            a_init_values = tf.convert_to_tensor(self.initial_scaling_factors, dtype="float32")
        return tf.Variable(initial_value=a_init_values, trainable=True)

    def build(self, input_shape):
        self.input_dims = self._create_input_dimensions(input_shape)
        self.output_dims = self.output_dim if self.output_dim else self.input_dims
        self.centers = self._create_centers(input_shape)
        self.a = self._create_scaling_factors(input_shape)
        super(FuzzifyLayer, self).build(input_shape)

    def call(self, x):
        x = self.expand(x, self.output_dims, axis=-1)
        c = self.expand_through(self.centers, self.input_dims)
        a = self.expand_through(self.scaling_factors, self.input_dims)
        xc = self.calc_membership(x, c, a)
        return xc

    @classmethod
    def expand(cls, val, dim, *, axis=0):
        return K.repeat_elements(K.expand_dims(val, axis), dim, axis)

    @classmethod
    def expand_through(cls, val, dims):
        for dim in dims:
            val = cls.expand(val, dim)
        return val

    @classmethod
    def calc_membership(cls, x, c, a):
        return K.exp(-K.sum(K.square((x - c) / (2 * a)), axis=-2, keepdims=False))

    # def compute_output_shape(self, input_shape):
    #     return tuple(input_shape[:-1]) + (self.output_dim,)


class DefuzzifyLayer(Layer, FuzzyOperations):

    def __init__(self,
            output_dim: Shape = None,
            initial_rules_outcomes=None,
            input_shape: tuple[int, ...] = None,
            input_dim: int = None,
            **kwargs
        ):
        if not input_shape and input_dim:
            input_shape = (input_dim,)

        self.initial_rules_outcomes = initial_rules_outcomes
        if input_shape:
            self.input_dims = list(input_shape)[:-1:-1]
            self.output_dims = output_dim if output_dim else self.input_dims
            self.rules_outcome = self._create_rules_outcome
        super(DefuzzifyLayer, self).__init__(**kwargs)

    def _create_rules_outcome(self, input_shape):
        if self.initial_rules_outcomes is None:
            outcomes_init_values = tf.random_uniform_initializer()(shape=(input_shape[-1], self.output_dims), dtype="float32")
        else:
            outcomes_init_values = tf.convert_to_tensor(self.initial_rules_outcomes, dtype="float32")
        return tf.Variable(initial_value=outcomes_init_values, trainable=True)

    def build(self, input_shape):
        self.input_dims = list(input_shape)[:-1:-1]
        self.output_dims = self.output_dims if self.output_dims else self.input_dims
        self.rules_outcome = self._create_rules_outcome
        super(DefuzzifyLayer, self).build(input_shape)

    def __call__(self, x):
        x = self.expand(x, self.output_dims, axis=-1)
        rules_outcome = self.expand_through(self.rules_outcome, self.input_dims)
        xc = self.defuzify(x, rules_outcome)
        return xc

    def defuzzify(self, x, rules_outcome):
        return K.sum((x * rules_outcome), axis=-2, keepdims=False)

    # def compute_output_shape(self, input_shape):
    #     return tuple(input_shape[:-1]) + (self.output_dim,)
    #
    # def get_config(self):
    #     return {"rules_outcome": self.rules_outcome.numpy()}


class FuzzyPooling(Layer):
    """Custom layer for Type-2 Fuzzy logic based pooling"""
    def __init__(self, kernel: int = 3, stride: int = 1, channels: int = None, dims: Shape = None):
        super().__init__()
        self.pool = kernel
        self.stride = stride
        self.n = kernel**2
        self.h = (self.n + 1)//2
        self.channels = channels
        self.dims = dims

    @property
    def rows(self) -> int:
        return self.dims[0]

    @property
    def cols(self) -> int:
        return self.dims[1]

    def build(self, input_shape):
        if not self.channels:
            self.channels = input_shape[0]
        if not self.dims:
            self.dims = input_shape[1:]

    def membership(self, x):
        h = self.h
        kmm = x.mean(dim=-1, keepdim=True)

        for k in range(h-1):
            start = h-k-1
            end = h+k+1 if self.pool % 2 == 0 else h+k
            x_partial_mean = tf.reduce_mean(x[..., start:end], axis=-1, keepdims=True)
            kmm = tf.concat([x_partial_mean, kmm], axis=-1)

        v_avg = kmm.mean(dim=-1, keepdim=True)
        return kmm, v_avg

    def calculate_fuzzy_variance(self, x, v_avg, eps=1e-4):
        h = self.h
        omega = abs(x - v_avg)
        sigma = omega.mean(dim=-1, keepdim=True)

        for k in range(h-1):
            start = (h-k-1)
            end = h+k if h % 2 == 1 else h+k+1
            partial_mean = tf.reduce_mean(x[..., start:end], axis=-1, keepdims=True)
            sigma = concatenate([partial_mean, sigma], axis=-1)
        return sigma + eps

    def __call__(self, x):
        batch_size = x.shape[0]
        self.output_size = self.row // self.pool

        # Since the method does not differentiate between two channels of a single image any more that it
        # differentiates between two channels of two separate images, it is beneficial to fold the first
        # two dimensions, ie, batch_size and channels together.

        x = tf.reshape(x, [batch_size * self.channels, *self.dims], name=None)
        x = tf.image.extract_patches(x,
                             sizes=[1, self.pool, self.pool, 1],
                             strides=[1, self.stride, self.stride, 1],
                             rates=[1, 1, 1, 1],
                             padding='VALID')
        x = tf.reshape(x, [-1, self.channels, self.dims[0], self.n * self.dims[1]])

        kmm, v_avg = self.membership(x)
        sigma = self.var_vec(x, v_avg)
        avg_pi, thresh = self.delta(x, kmm, sigma)

        pooled = tf.zeros([self.batch * self.channels, self.output_size, self.output_size, 1])

        # Conditions
        mask_primary = tf.greater_equal(avg_pi[..., self.h-1], thresh)
        s_condition = tf.less(sigma[..., self.h-1], 0.001)
        mask_secondary = tf.reduce_sum(tf.cast(~mask_primary & s_condition, tf.float32), axis=-1, keepdims=True)
        mask_noisy = ~(mask_primary | mask_secondary)

        # pooled operations
        pooled = tf.where(mask_primary, tf.reduce_mean(x, axis=-1, keepdims=True), pooled)
        pooled = tf.where(mask_secondary, v_avg, pooled)

        count = tf.reduce_sum(tf.cast(mask_noisy, tf.float32)).numpy()  # Count
        region = tf.boolean_mask(x, mask_noisy[..., tf.newaxis])  # Region
        g = tf.boolean_mask(avg_pi, mask_noisy[..., tf.newaxis])  # g
        denoised = tf.reduce_sum(tf.multiply(g, region), axis=-1, keepdims=True) / tf.reduce_sum(g, axis=-1, keepdims=True)
        denoised = tf.reshape(denoised, [count])  # Denoised

        indices = tf.where(mask_noisy)
        pooled = tf.tensor_scatter_nd_update(pooled, indices, denoised)

        pooled = tf.reshape(pooled, [self.batch, self.channels, self.output_size, self.output_size])
        pooled = tf.identity(pooled)

        return pooled

    def delta(self, x, kmm, sigma):
        h = self.h
        n = self.n

        # All the next 4 Tensors are of shape (batch*channels, output_size, output_size, h, n)
        xrep = x.repeat(1, 1, 1, h).view(self.batch*self.channels, self.output_size, self.output_size, h, n)
        kmmrep = kmm.repeat(1, 1, 1, n).view(self.batch*self.channels, self.output_size, self.output_size, n, h).transpose(3, 4)
        sigmarep = sigma.repeat(1, 1, 1, n).view(self.batch*self.channels, self.output_size, self.output_size, n, h).transpose(3, 4)
        pi = tf.exp(-0.5 * tf.square((xrep - kmmrep) / sigmarep))

        max_values = tf.reduce_max(pi, axis=3, keepdims=False)
        thresh = tf.reduce_min(max_values, axis=3, keepdims=True)
        avg_pi = tf.reduce_mean(pi, axis=3, keepdims=False)
        return avg_pi, thresh



class FuzzyLayer(Layer):
    def __init__(self, fuzzify: Layer, defuzzify: Layer):
        super().__init__()
        self.fuzzify = fuzzify
        self.defuzzify = defuzzify

    def __call__(self, x):
        x = self.fuzzify(x)
        x = self.defuzzify(x)
        return x