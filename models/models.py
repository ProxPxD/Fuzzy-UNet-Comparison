from dataclasses import dataclass
from general_unet import UNet
from models.fuzzy import DefuzzyLayer, FuzzyLayer, FuzzyPooling


class ModelFactory:
    @classmethod
    def build(cls, fuzzy_link: bool, fuzzy_pooling: bool, **layers_kwargs):
        kwargs = {}
        if fuzzy_link:  # TODO set only on the first as it was showed the best
            kwargs['before_link'] = FuzzyLayer()
            kwargs['after_link'] = DefuzzyLayer()
        if fuzzy_pooling:  # TODO: consider setting only in some places?
            kwargs['pooling'] = FuzzyPooling()
        return UNet(**kwargs)
