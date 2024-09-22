from itertools import product

from keras import Model
from keras.src.layers import MaxPooling2D

from custom_models.fuzzy import FuzzyPooling, ConvFuzzyLayer
from custom_models.general_components import UNet, CNN
from more_itertools import repeatfunc

model_space = ('unet', 'cnn')
bool_space = (True, False)


def gen_param_set(param_space):
    return (dict(zip(param_space.keys(), params)) for params in product(*param_space.values()))


class ModelFactory:
    @classmethod
    def get_models_to_analyze(cls, space: dict, **kwargs) -> dict[str, Model]:
        return {
            ModelFactory.get_name(**params): ModelFactory.build(**params, **kwargs)
            for params in gen_param_set(space)
        }

    @classmethod
    def get_name(cls,
            model: str,
            fuzzy_pooling: bool,
            fuzzy_layer: bool,
            **layers_kwargs
        ) -> str:
        modifier = 'Fuzzy' if fuzzy_layer else 'Crisp'
        name = model.upper()
        pooling = 'Fuzzy' if fuzzy_pooling else 'Crisp'
        return f'{modifier} Layered {name} with {pooling} Pooling'

    @classmethod
    def build(cls,
            model: str,
            fuzzy_pooling: bool,
            fuzzy_layer: bool,
            depth: int = 3,
            **layers_kwargs
        ) -> Model:
        match model.lower():
            case 'unet': return cls.build_unet(fuzzy_pooling, fuzzy_layer, depth, **layers_kwargs)
            case 'cnn': return cls.build_cnn(fuzzy_pooling, fuzzy_layer, **layers_kwargs)
            case _: raise ValueError(f'Model {model} is not supported, try one of {model_space}')

    @classmethod
    def build_unet(cls,
            fuzzy_pooling: bool,
            fuzzy_layer: bool,
            depth: int = 3,
            **layers_kwargs
    ) -> UNet:
        match (fuzzy_layer, fuzzy_pooling):
            case (True, True): raise NotImplementedError
            case (True, False): raise NotImplementedError
            case (False, True): raise NotImplementedError
            case (False, False): raise NotImplementedError
            case _: raise ValueError(f'{fuzzy_layer, fuzzy_pooling = } are not supported!')

    @classmethod
    def build_cnn(cls,
            fuzzy_pooling: bool,
            fuzzy_layer: bool,
            **layers_kwargs
    ) -> CNN:
        match (fuzzy_layer, fuzzy_pooling):
            case (bool(), bool()): pass
            case _: raise ValueError(f'{fuzzy_layer, fuzzy_pooling = } are not supported!')

        after_conv_layer = ConvFuzzyLayer() if fuzzy_layer else None
        dense_units = (None if fuzzy_layer else 64) and 64 or 64
        pooling = FuzzyPooling if fuzzy_pooling else None
        cnn = CNN(
            pooling=pooling,
            dense_units=dense_units,
            after_conv_layer=after_conv_layer,
            flatten=True,
            **layers_kwargs
        )
        return cnn


space = {
    'model': model_space and ('cnn', ),
    'fuzzy_layer': bool_space,
    'fuzzy_pooling': bool_space or [False],
    'depth': [3],
}
