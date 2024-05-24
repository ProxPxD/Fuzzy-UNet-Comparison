from itertools import repeat
from typing import Optional, Sequence, Iterable, Reversible, Tuple, Union

from keras import Sequential
from keras.layers import Layer, Conv2D, MaxPooling2D, UpSampling2D, concatenate, ReLU
from more_itertools import interleave, pairwise, last, padded

from models.utils import repeat_last_up_to, to_list
import keras.backend as K

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
    def shapes(self) -> Union[Iterable[IntPair], Sequence[IntPair]]:  # 64, 128, ...
        return list(pairwise(map(lambda d: 64*2**d, range(self.depth+1))))

    @property
    def last_shape(self):
        return last(self.shapes)[1]


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
            case 1: return self.layers[0][x]
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
            ) for input_size, output_size in self.shapes
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
            ) for input_size, output_size in reversed(self.shapes)
        ]

    def call(self, x, from_links):
        for decode, from_link in zip(self.decoder_units, reversed(from_links)):
            x = decode(x, from_link)
        return x


class Linkage(IName, IDepth, Layer):
    def __init__(self, links: LayerOrMore = Layer(), depth=4):
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
            links: LayerOrMore = Layer(),
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

