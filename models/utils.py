from typing import Sequence

from more_itertools import repeat_last, take


def to_list(elem_or_more):
    return [elem_or_more] if not isinstance(elem_or_more, Sequence) or isinstance(elem_or_more, str) else list(elem_or_more)


def repeat_last_up_to(elems, up_to=None):
    n_channels = [elems] if not isinstance(elems, Sequence) else up_to
    all_n_channels = repeat_last(n_channels)
    return take(up_to, all_n_channels) if up_to is not None else all_n_channels



    # kernels = self._init_kernels(kernel, kwargs)

    # def _init_kernels(self, kernel, kwargs):  # TODO: move to interface to allow decoder to use it
    #     if isinstance(kernel, Sequence) and len(kernel) != kwargs.get('n_conv_layers', 2):
    #         raise ValueError
    #     kernels = repeat(kernel) if isinstance(kernel, int) else kernel
    #     return kernels