from itertools import chain, product

from keras import Layer
from more_itertools import repeatfunc

from general_unet import UNet
from fuzzy import DefuzzyLayer, FuzzyLayer, FuzzyPooling


def gen_param_set(param_space):
    return (dict(zip(param_space.keys(), params)) for params in product(*param_space.values()))


class ModelFactory:
    @classmethod
    def build(cls,
            fuzzy_link: bool,
            fuzzy_pooling: bool,
            n_fuzzy_layers: int = 1,
            **layers_kwargs
        ):
        kwargs = {}
        if fuzzy_link:
            kwargs['before_link'] = chain(repeatfunc(FuzzyLayer,   n_fuzzy_layers), Layer())
            kwargs['after_link']  = chain(repeatfunc(DefuzzyLayer, n_fuzzy_layers), Layer())
        if fuzzy_pooling:  # TODO: consider setting only in some places?
            kwargs['pooling'] = FuzzyPooling()
        return UNet(**kwargs)

    @classmethod
    def get_name(cls,
            fuzzy_link: bool,
            fuzzy_pooling: bool,
            n_fuzzy_layers: int = 1,
            **layers_kwargs
        ):
        if not fuzzy_link and not fuzzy_pooling:
            return 'Crisp'
        if fuzzy_link and not fuzzy_pooling:
            return 'Fuzzy Layer'
        if not fuzzy_link and fuzzy_pooling:
            return 'Fuzzy Pooling'
        if fuzzy_link and fuzzy_pooling:
            return 'Full Fuzzy'
        return 'Unnamed'

    @classmethod
    def get_models_to_analyze(cls, space):
        return {
            ModelFactory.get_name(*params): ModelFactory.build(*params)
            for params in gen_param_set(space)
        }


bool_space = [True, False]

space = {
    'fuzzy_link': bool_space,
    'fuzzy_pooling': bool_space,
    'n_fuzzy_layers': [1],
}




