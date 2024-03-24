from itertools import repeat, chain

from keras import Layer
from more_itertools import repeatfunc

from general_unet import UNet
from models.fuzzy import DefuzzyLayer, FuzzyLayer, FuzzyPooling


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


bool_space = [True, False]

space = {
    'fuzzy_link': bool_space,
    'fuzzy_pooling': bool_space,
    'n_fuzzy_layers': [1],
}
