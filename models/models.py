from itertools import product, repeat, starmap

import keras
from keras.layers import Layer
from more_itertools import repeatfunc, take
from toolz import apply

from models.fuzzy import DefuzzifyLayer, FuzzifyLayer, FuzzyPooling, FuzzyLayer
from models.general_components import UNet, Link


def gen_param_set(param_space):
    return (dict(zip(param_space.keys(), params)) for params in product(*param_space.values()))


class ModelFactory:
    @classmethod
    def build(cls,
            fuzzy_pooling: bool,
            outer_fuzzy_later: bool,
            n_fuzzy_layers: int,
            **layers_kwargs
        ):
        kwargs = {}
        if outer_fuzzy_later:
            kwargs['outer_link'] = Link([FuzzyLayer(), Layer()])
        if n_fuzzy_layers:
            to_link = zip(repeatfunc(FuzzifyLayer), repeatfunc(DefuzzifyLayer))
            to_apply = zip(repeat(keras.Sequential), to_link)
            with_extra = zip(starmap(apply, to_apply), repeatfunc(Layer))
            kwargs['links'] = take(n_fuzzy_layers, with_extra)
        if fuzzy_pooling:  # TODO: consider setting only in some places?
            kwargs['pooling'] = FuzzyPooling()
        return UNet(**kwargs)

    @classmethod
    def get_name(cls,
            fuzzy_pooling: bool,
            outer_fuzzy_later: bool,
            n_fuzzy_layers: int,
            **layers_kwargs
        ):
        match (n_fuzzy_layers, fuzzy_pooling, outer_fuzzy_later):
            case (0, False, False): name = 'Crisp'
            case (0, False, True): name = 'Outer Fuzzy Layer'
            case (_, False, False): name = 'Inner Fuzzy Layer'
            case (_, False, True): name = 'Inner and Outer Fuzzy Layer'
            case (0, True, False): name = 'Fuzzy Pooling'
            case (0, True, True): name = 'Outer Fuzzy Layer with Fuzzy Pooling'
            case (_, True, False): name = 'Inner Fuzzy Layer with Fuzzy Pooling'
            case (_, True, True): name = 'Full Fuzzy'
            case _: name = 'Unnamed'
        return name

    @classmethod
    def get_models_to_analyze(cls, space: dict) -> dict[str, UNet]:
        return {
            ModelFactory.get_name(**params): ModelFactory.build(**params)
            for params in gen_param_set(space)
        }


binary_space = [0, 1]
bool_space = [True, False]

space = {
    'n_fuzzy_layers': binary_space,
    'outer_fuzzy_later': bool_space,
    'fuzzy_pooling': bool_space,
}




