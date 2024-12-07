from itertools import product

from keras.layers import Layer

from custom_models.fuzzy import FuzzyPooling, FuzzyLayer
from custom_models.general_components import UNet, Link, IDepth
from more_itertools import repeatfunc


def gen_param_set(param_space):
    return (dict(zip(param_space.keys(), params)) for params in product(*param_space.values()))


class ModelFactory:
    @classmethod
    def build(cls,
            fuzzy_pooling: bool,
            outer_fuzzy_later: bool,
            n_fuzzy_layers: int,
            depth: int = 3,
            **layers_kwargs
        ):
        kwargs = {}
        if outer_fuzzy_later:
            kwargs['outer_link'] = Link([FuzzyLayer(input_dim=IDepth(depth=depth).last_shape), Layer()])
        if n_fuzzy_layers:
            fuzzy_layers = (FuzzyLayer(input_dim=input_dim) for input_dim in IDepth(depth=depth).shapes)
            links = [Link([fuzzy_layer, layer]) for fuzzy_layer, layer in zip(fuzzy_layers, repeatfunc(Layer))]
            kwargs['links'] = links
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
    'fuzzy_pooling': [False],
    'depth': [3],
}




