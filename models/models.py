from itertools import chain, product, repeat

import keras
from keras.layers import Layer
from more_itertools import repeatfunc, take
from toolz import apply

from models.fuzzy import DefuzzifyLayer, FuzzifyLayer, FuzzyPooling, FuzzyLayer
from models.general_components import UNet, Link
from itertools import starmap

from models.utils import fill_func_upto


def gen_param_set(param_space):
    return (dict(zip(param_space.keys(), params)) for params in product(*param_space.values()))


class ModelFactory:
    @classmethod
    def build(cls,
            fuzzy_pooling: bool,
            n_fuzzy_layers: int = 1,
            **layers_kwargs
        ):
        kwargs = {}
        if n_fuzzy_layers:
            to_link = zip(repeatfunc(FuzzifyLayer), repeatfunc(DefuzzifyLayer))
            to_apply = zip(repeat(keras.Sequential), to_link)
            kwargs['links'] = take(n_fuzzy_layers, starmap(apply, to_apply))
        if fuzzy_pooling:  # TODO: consider setting only in some places?
            kwargs['pooling'] = FuzzyPooling()
        return UNet(**kwargs)

    @classmethod
    def get_name(cls,
            fuzzy_pooling: bool,
            n_fuzzy_layers: int = 1,
            **layers_kwargs
        ):
        if not n_fuzzy_layers and not fuzzy_pooling:
            return 'Crisp'
        if n_fuzzy_layers and not fuzzy_pooling:
            return 'Fuzzy Layer'
        if not n_fuzzy_layers and fuzzy_pooling:
            return 'Fuzzy Pooling'
        if n_fuzzy_layers and fuzzy_pooling:
            return 'Full Fuzzy'
        return 'Unnamed'

    @classmethod
    def get_models_to_analyze(cls, space: dict) -> dict[str, UNet]:
        return {
            ModelFactory.get_name(**params): ModelFactory.build(**params)
            for params in gen_param_set(space)
        }


bool_space = [True, False]

space = {
    'n_fuzzy_layers': [0, 1],
    'fuzzy_pooling': bool_space,
}




