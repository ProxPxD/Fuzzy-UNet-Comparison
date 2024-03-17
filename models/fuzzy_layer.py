from keras import backend as K
from tensorflow import keras
import tensorflow as tf

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


class FuzzyLayer(keras.layers.Layer, FuzzyOperations):

    def __init__(self,
            output_dim,
            initial_centers=None,
            initial_scaling_factors=None,
            input_shape: tuple[int, ...] = None,
            input_dim: int = None,
            **kwargs
        ):
        if not input_shape and input_dim:
            input_shape = (input_dim,)
        self.output_dim = output_dim
        self.initial_centers = initial_centers
        self.initial_sigmas = initial_scaling_factors
        if input_shape:
            self.input_dimensions = self._create_input_dimensions(input_shape)
            self.centers = self._create_centers(input_shape)
            self.a = self._create_scaling_factors(input_shape)
        super(FuzzyLayer, self).__init__(**kwargs)

    @classmethod
    def _create_input_dimensions(cls, input_shape: Shape):
        return list(input_shape)[:-1:-1]

    def _create_centers(self, input_shape: Shape):
        if self.initial_centers is None:
            c_init_values = tf.random_uniform_initializer(-1, 1)(shape=(input_shape[-1], self.output_dim), dtype="float32")
        else:
            c_init_values = tf.convert_to_tensor(self.initial_centers, dtype="float32")
        return tf.Variable(initial_value=c_init_values, trainable=True)

    def _create_scaling_factors(self, input_shape: Shape):
        if self.initial_scaling_factors is None:
            a_init_values = tf.ones_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32")
        else:
            a_init_values = tf.convert_to_tensor(self.initial_scaling_factors, dtype="float32")
        return tf.Variable(initial_value=a_init_values, trainable=True)

    def build(self, input_shape):
        self.input_dimensions = self._create_input_dimensions(input_shape)
        self.centers = self._create_centers(input_shape)
        self.a = self._create_scaling_factors(input_shape)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, x):
        x = self.expand(x, self.output_dim, axis=-1)
        c = self.expand_through(self.centers, self.input_dimensions)
        a = self.expand_through(self.scaling_factors, self.input_dimensions)
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


class DefuzzyLayer(keras.layers.Layer, FuzzyOperations):

    def __init__(self,
                 output_dim,
                 initial_rules_outcomes=None,
                  input_shape: tuple[int, ...] = None,
            input_dim: int = None,
            **kwargs
        ):
        if not input_shape and input_dim:
            input_shape = (input_dim,)

        self.output_dim = output_dim
        self.initial_rules_outcomes = initial_rules_outcomes
        if input_shape:
            self.input_dimensions = list(input_shape)[:-1:-1]
            self.rules_outcome = self._create_rules_outcome
        super(DefuzzyLayer, self).__init__(**kwargs)

    def _create_rules_outcome(self, input_shape):
        if self.initial_rules_outcomes is None:
            outcomes_init_values = tf.random_uniform_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32")
        else:
            outcomes_init_values = tf.convert_to_tensor(self.initial_rules_outcomes, dtype="float32")
        return tf.Variable(initial_value = outcomes_init_values, trainable=True)

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]
        self.rules_outcome = self._create_rules_outcome
        super(DefuzzyLayer, self).build(input_shape)

    def call(self, x):
        x = self.expand(x, self.output_dim, axis=-1)
        rules_outcome = self.expand_through(self.rules_outcome, self.input_dimensions)
        xc = self.defuzify(x, rules_outcome)
        return xc

    def defuzzify(self, x, rules_outcome):
        return K.sum((x * rules_outcome), axis=-2, keepdims=False)

    # def compute_output_shape(self, input_shape):
    #     return tuple(input_shape[:-1]) + (self.output_dim,)
    #
    # def get_config(self):
    #     return {"rules_outcome": self.rules_outcome.numpy()}
