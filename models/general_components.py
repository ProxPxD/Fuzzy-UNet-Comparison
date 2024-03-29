from itertools import repeat
from typing import Optional, Sequence, Iterable, Reversible, Tuple, Union

from keras import Sequential
from keras.layers import Layer, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers.activations import ReLU
from more_itertools import interleave, pairwise, last

from models.utils import repeat_last_up_to

IntPair = Tuple[int, int]
LayerOrMore = Layer | Sequence[Layer]


class IName:
    def __init__(self, name: Optional[str] = None, *args, **kwargs):
        self.name = name or self.__class__.__name__
        super(*args, **kwargs)


class IDepth:
    def __init__(self, depth: int, *args, **kwargs):
        self.depth: int = depth
        super().__init__(*args, **kwargs)

    @property
    def shapes(self) -> Union[Iterable[IntPair], Reversible[IntPair]]:  # 64, 128, ...
        return pairwise(map(lambda d: 64*2**d, range(self.depth+1)))

    @property
    def last_shape(self):
        return last(self.shapes)[1]


class MultiConv(Layer, IName):
    def __init__(self,
            n_channels: int | Sequence[int],
            n_conv_layers: int = 2,
            kernel: int = 3,
            batch_norm: Optional[Layer] = None,
            activation: Layer = ReLU(),
            **kwargs,
        ):
        super().__init__(**kwargs)
        all_n_channels = repeat_last_up_to(n_channels, n_conv_layers)
        self._layers = Sequential(list(interleave(
            (Conv2D(curr_n_channels, kernel, bias_flag=not batch_norm, **kwargs) for curr_n_channels in all_n_channels),
            repeat(batch_norm) if batch_norm else [],
            repeat(activation),
        )))

    def __call__(self, inputs):
        return self._layers(inputs)


class EncoderUnit(Layer):
    def __init__(self,
            pooling: Layer = MaxPooling2D(),
            before_link: Layer = Layer(),
            **kwargs
        ):
        super().__init__(**kwargs)
        self.mc = MultiConv(**kwargs)
        self.pooling = pooling
        self.before_link = before_link

    def __call__(self, inputs):
        convoluted = self.mc(inputs)
        x_encoded = self.pooling(convoluted)
        postprocessed = self.before_link(convoluted)
        return x_encoded, postprocessed


class DecoderUnit(Layer):
    def __init__(self,
            up_sampling: Layer = UpSampling2D(),
            after_link: Layer = Layer(),
            **kwargs
        ):
        super().__init__(**kwargs)
        self.up_sample = up_sampling
        self.mc = MultiConv(**kwargs)
        self.after_link = after_link

    def __call__(self, x, to_link):
        up_sampled = self.up_sample(x)
        preprocessed = self.after_link(to_link)
        cated = concatenate([up_sampled, preprocessed], axis=3)  # TODO: check axis
        y = self.mc(cated)
        return y


class Encoder(Layer, IDepth, IName):
    def __init__(self,
            pooling: Layer = MaxPooling2D(),
            depth: int = 4,
            kernel: int | Sequence[int] = 3,  # INFO: right now, sequence not supported
            before_link: LayerOrMore = Layer(),
            **kwargs
        ):
        super().__init__(depth=depth, **kwargs)

        self.encoder_units = [
            EncoderUnit(
                n_channels=input_size,
                pooling=pooling,
                kernel=kernel,
                before_link=curr_before_link,
                **kwargs
            ) for (input_size, output_size), curr_before_link in zip(self.shapes, repeat_last_up_to(before_link))
        ]

    def __call__(self, inputs):
        curr_x = inputs
        x_to_cats = []
        for encode in self.encoder_units:
            curr_x, x_to_cat = encode(curr_x)
            x_to_cats.append(x_to_cat)
        return curr_x, x_to_cats


class Decoder(Layer, IDepth, IName):
    def __init__(self,
            up_sampling: Layer = UpSampling2D(),
            depth: int = 4,
            after_link: LayerOrMore = Layer(),
            **kwargs
        ):
        super().__init__(depth=depth, **kwargs)
        self.decoder_units = [
            DecoderUnit(
                n_channels=(output_size, input_size),
                up_sampling=up_sampling,
                after_link=curr_after_link,
                **kwargs
            ) for (input_size, output_size), curr_after_link in reversed(zip(self.shapes, repeat_last_up_to(after_link, self.depth)))
        ]

    def __call__(self, x, to_cats):
        for decode, to_cat in zip(self.decoder_units, reversed(to_cats)):
            x = decode(x, to_cat)
        return x


class BottleNeck(Layer, IName, IDepth):
    def __init__(self, depth: int = 4, **kwargs):
        super().__init__(depth=depth, **kwargs)
        self.mc = MultiConv(n_channels=self.last_shape, *kwargs)

    def __call__(self, x):
        return self.mc(x)


class LinkConjunct(Layer, IName, IDepth):
    def __init__(self,
            link_layer: LayerOrMore = Layer(),
            depth: int = 3,
            **kwargs
        ):
        super().__init__(depth=depth, **kwargs)
        self.link_layers = list(repeat_last_up_to(link_layer, self.depth + 1))
        repeat_last_up_to(link_layer, self.depth + 1)

    def __call__(self, outer, to_cats):
        return self.link_layers[0](outer), [link_layer(to_cat) for link_layer, to_cat in zip(self.link_layers, to_cats)]


class UNet(Layer, IDepth):
    def __init__(self,
            depth: int = 3,
            n_conv_layers: int = 2,
            pooling: Layer = MaxPooling2D(),
            up_sampling: Layer = UpSampling2D(),
            activation: Layer = ReLU(),
            link_layers: LayerOrMore = Layer(),
            **kwargs
        ):
        super().__init__(depth=depth, **kwargs)
        self.encode = Encoder(depth=depth,         n_conv_layers=n_conv_layers, activation=activation, pooling=pooling,         **kwargs)
        self.decode = Decoder(depth=depth,         n_conv_layers=n_conv_layers, activation=activation, up_sampling=up_sampling, **kwargs)
        # self.conjunct = LinkConjunct(depth=depth,                                                      link_layers=link_layers, **kwargs)
        self.bottle_neck = BottleNeck(depth=depth, n_conv_layers=n_conv_layers, activation=activation,                          **kwargs)

    def __call__(self, input):
        x, to_cats = self.encode(input)
        # x, to_links = self.encode(input)
        # outer, to_cats = self.conjunct(input, to_links)  # TODO: what to do with the outer as for the paper of Galina et al.
        x = self.bottle_neck(x)
        x = self.decode(x, to_cats)
        return x


