from keras.layers import Layer, MaxPooling2D, UpSampling2D, ReLU

from models.general_components import Encoder, Decoder, BottleNeck


class UNet(Layer):
    def __init__(self,
            depth: int = 4,
            n_conv_layers: int = 2,
            pooling: Layer = MaxPooling2D(),
            up_sampling: Layer = UpSampling2D(),
            activation: Layer = ReLU(),
            **kwargs
        ):
        super().__init__(depth=depth, **kwargs)
        self.encode = Encoder(depth=depth,         n_conv_layers=n_conv_layers, activation=activation, pooling=pooling,         **kwargs)
        self.decode = Decoder(depth=depth,         n_conv_layers=n_conv_layers, activation=activation, up_sampling=up_sampling, **kwargs)
        self.bottle_neck = BottleNeck(depth=depth, n_conv_layers=n_conv_layers, activation=activation,                          **kwargs)

    def __call__(self, input):
        x, to_cats = self.encode(input)
        x = self.bottle_neck(x)
        x = self.decode(x, to_cats)
        return x