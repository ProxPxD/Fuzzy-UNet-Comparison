import keras
import tensorflow as tf
from keras import layers, backend as K

from custom_models.fuzzy_layers.KerasFuzzy.layers.defuzzy_layer import DefuzzyLayer
from custom_models.fuzzy_layers.KerasFuzzy.layers.fuzzy_layer_2 import FuzzyLayer2

latent_dim = 3

mnist_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", padding="same")(mnist_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
z = layers.Dense(latent_dim, name="z")(x)
base_model = keras.Model(mnist_inputs, z)

print(f'pre base model: {mnist_inputs.shape = }')
x = base_model(mnist_inputs)
print(f'pre fuzzy: {x.shape = }')
x = FuzzyLayer2(10, name="fuzzy")(x)
print(f'pre defuzzy: {x.shape = }')
x = DefuzzyLayer(10, name="defuzzy")(x)
print(f'post defuzzy: {x.shape = }')
x = tf.keras.layers.Softmax()(x)
print(f'post softmax: {x.shape = }')
fuzzy_model = keras.Model(mnist_inputs, x)


x = tf.random.normal((28, 28, 1))
x_batch = tf.expand_dims(x, axis=0)
y = fuzzy_model(x_batch)
print(f'{y = }')
