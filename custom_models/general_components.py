import logging
from itertools import repeat
from typing import Optional, Sequence, Iterable, Tuple, Union

import keras.backend as K
from keras import Sequential, Model
from keras.layers import Layer, Conv2D, MaxPooling2D, UpSampling2D, ReLU, Lambda, Dense, Flatten
from more_itertools import interleave, pairwise, last, padded, repeatfunc
from toolz import compose_left as pipe

from custom_models.utils import repeat_last_up_to, to_list

IntPair = Tuple[int, int]
LayerOrMore = Layer | Sequence[Layer]


class IName:
    def __init__(self, name: Optional[str] = None, *args, **kwargs):
        super().__init__(name=name or self.__class__.__name__, *args, **kwargs)


class IDepth:
    def __init__(self, depth: int, *args, **kwargs):
        self.depth: int = depth
        super().__init__(*args, **kwargs)

    @property
    def shape_pairs(self) -> Union[Iterable[IntPair], Sequence[IntPair]]:
        return list(pairwise(self.shapes))

    @property
    def shapes(self) -> Union[Sequence[int], Iterable[int]]:  # 64, 128, ...
        return list(map(lambda d: 64*2**d, range(self.depth+1)))

    @property
    def last_shape(self):
        return last(self.shape_pairs)[1]


class MultiConv(IName, Layer):
    def __init__(self,
            n_channels: int | Sequence[int],
            n_conv_layers: int = 2,
            kernel: int = 3,
            batch_norm: Optional[Layer] = None,
            activation: Layer = ReLU(),
            **kwargs,
        ):
        super().__init__()
        all_n_channels = repeat_last_up_to(n_channels, n_conv_layers)
        self._layers = Sequential(list(interleave(
            (Conv2D(curr_n_channels, kernel, use_bias=not batch_norm, **kwargs) for curr_n_channels in all_n_channels),
            repeat(batch_norm) if batch_norm else [],
            repeat(activation),
        )))

    def call(self, inputs):
        return self._layers(inputs)


class EncoderUnit(Layer):
    def __init__(self,
            pooling: Layer = MaxPooling2D(),
            **kwargs
        ):
        super().__init__()
        self.mc = MultiConv(**kwargs)
        self.pooling = pooling

    def call(self, inputs):
        convoluted = self.mc(inputs)
        x_encoded = self.pooling(convoluted)
        return x_encoded, x_encoded


class DecoderUnit(Layer):
    def __init__(self,
            up_sampling: Layer = UpSampling2D(),
            **kwargs
        ):
        super().__init__()
        self.up_sample = up_sampling
        self.mc = MultiConv(**kwargs)

    def call(self, x, to_link):
        up_sampled = self.up_sample(x)
        # cated = concatenate([up_sampled, preprocessed], axis=3)  # TODO: check axis
        y = self.mc(up_sampled)
        return y


class Link(Layer, IName, IDepth):
    def __init__(self, layers: LayerOrMore = None):
        super().__init__()
        self.layers = to_list(layers or [])
        n = len(self.layers)
        self.ws = None if n < 2 else K.ones(n) / n

    def call(self, x):
        match len(self.layers):
            case 0: return x
            case 1: return self.layers[0](x)
            case _: return sum(layer(x)*w/w.sum() for layer, w in zip(self.layers, self.ws))


class Encoder(IDepth, IName, Layer):
    def __init__(self,
            pooling: Layer = MaxPooling2D(),
            depth: int = 4,
            kernel: int | Sequence[int] = 3,  # INFO: right now, sequence not supported
            **kwargs
        ):
        super().__init__(depth=depth)

        self.encoder_units = [
            EncoderUnit(
                n_channels=input_size,
                pooling=pooling,
                kernel=kernel,
                **kwargs
            ) for input_size, output_size in self.shape_pairs
        ]

    def call(self, inputs):
        curr_x = inputs
        encodeds = []
        for encode in self.encoder_units:
            curr_x, encoded = encode(curr_x)
            encodeds.append(encoded)
        return curr_x, encodeds


class Decoder(IDepth, IName, Layer):
    def __init__(self,
            up_sampling: Layer = UpSampling2D(),
            depth: int = 4,
            **kwargs
        ):
        super().__init__(depth=depth)
        self.decoder_units = [
            DecoderUnit(
                n_channels=(output_size, input_size),
                up_sampling=up_sampling,
                **kwargs
            ) for input_size, output_size in reversed(self.shape_pairs)
        ]

    def call(self, x, from_links):
        for decode, from_link in zip(self.decoder_units, reversed(from_links)):
            x = decode(x, from_link)
        return x


class Linkage(IName, IDepth, Layer):
    def __init__(self, links: LayerOrMore = Lambda(lambda x: x), depth=4):
        super().__init__(depth=depth)
        self.links = [Link(layers=link) for link in padded(to_list(links), n=depth)]

    def call(self, xs):
        return [link(x) for link, x in zip(self.links, xs)]


class BottleNeck(IName, IDepth, Layer):
    def __init__(self, depth: int = 4, **kwargs):
        super().__init__(depth=depth)
        self.mc = MultiConv(n_channels=self.last_shape)

    def call(self, x):
        return self.mc(x)


class UNet(IDepth, Layer):
    def __init__(self,
            depth: int = 3,
            n_conv_layers: int = 2,
            pooling: Layer = MaxPooling2D(),
            up_sampling: Layer = UpSampling2D(),
            activation: Layer = ReLU(),
            links: LayerOrMore = Lambda(lambda x: x),
            outer_link: Layer = None,
            **kwargs
        ):
        super().__init__(depth=depth, **kwargs)
        self.encode = Encoder(depth=depth,         n_conv_layers=n_conv_layers, activation=activation, pooling=pooling,         **kwargs)
        self.decode = Decoder(depth=depth,         n_conv_layers=n_conv_layers, activation=activation, up_sampling=up_sampling, **kwargs)
        self.linkage = Linkage(depth=depth, links=links)
        self.outer_link = outer_link
        self.bottle_neck = BottleNeck(depth=depth, n_conv_layers=n_conv_layers, activation=activation,                          **kwargs)

    def call(self, x):
        initial_x = x
        x, encoded = self.encode(x)
        link_mapped = self.linkage(encoded)
        x = self.bottle_neck(x)
        x = self.decode(x, link_mapped)
        if self.outer_link:
            x = self.outer_link(initial_x)
        return x


class CNN(Model):
    def __init__(self,
            input_shape: tuple,
            n_classes: int,

            all_n_channels: tuple[int, ...],
            kernel_size=(3, 3),
            pooling: Layer = None,
            conv_activation='relu',

            after_conv_layer: Layer = None,

            flatten: Layer | bool = None,

            dense_units=None,
            dense_activation='relu',

            output_layer=None,
            output_activation='softmax',
        ):
        super().__init__()
        pooling = pooling or MaxPooling2D
        self.conv = self._create_conv_layers(input_shape, all_n_channels, conv_activation, kernel_size, pooling)
        self.after_conv_layer = after_conv_layer or Lambda(lambda x: x)
        self.flatten = Flatten() if flatten is True else flatten or Lambda(lambda x: x)
        self.dense = self._create_dense_layers(dense_units, dense_activation)
        self.output_layer = output_layer or Dense(n_classes, activation=output_activation)

    def _create_conv_layers(self, input_shape, all_n_channels: tuple[int, ...], conv_activation: str, kernel_size: int, pooling: Layer) -> Layer:
        conv_layers = [
            Conv2D(all_n_channels[0], kernel_size, activation=conv_activation, input_shape=input_shape),
            *[Conv2D(n_channels, kernel_size, activation=conv_activation)
              for n_channels in all_n_channels[1:]]
        ]
        return Sequential(list(interleave(conv_layers, repeatfunc(pooling))))

    def _create_dense_layers(self, dense_units, activation: str) -> Layer:
        if not dense_units:
            return Lambda(lambda x: x)
        if isinstance(dense_units, int):
            dense_units = (dense_units, )
        if not isinstance(dense_units, Iterable):
            raise ValueError('dense_units should be int or an iterable')
        dense = Sequential([
            Dense(units, activation=activation) for units in dense_units
        ])
        return dense

    def call(self, x):
        for name in ['conv', 'after_conv_layer', 'flatten', 'dense', 'output_layer']:
            logging.info(f'CNN: {name} = {getattr(self, name)}')
        return pipe(
            self.conv,
            lambda x: logging.debug(f'CNN AFTER conv: {x.shape}') or x,
            self.after_conv_layer,
            lambda x: logging.debug(f'CNN AFTER after_conv_layer: {x.shape}') or x,
            self.flatten,
            lambda x: logging.debug(f'CNN AFTER flatten: {x.shape}') or x,
            self.dense,
            lambda x: logging.debug(f'CNN AFTER dense: {x.shape}') or x,
            self.output_layer,
        )(x)
