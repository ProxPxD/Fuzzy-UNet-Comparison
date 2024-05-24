from itertools import takewhile

import cv2
import numpy as np
from keras.utils import Sequence
from more_itertools import random_permutation, split_into
from pydash import chain as c, flow

import utils
from constants import Params, Paths


class CamSeqSequence(Sequence):
    def __init__(self, path_tuples, batch_size: int = 5, normalize=lambda args: args):
        self.path_tuples = path_tuples
        self.normalize = normalize
        self.image_transposition = (2, 0, 1)
        self.index = 0
        self.batch_size = batch_size

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, index):
        load = flow(str, cv2.imread)
        # load = compose_left(str, cv2.imread, self.normalize)
        img, mask = c(self.path_tuples[index]).map(load).apply(self.normalize).value()
        img = img.transpose(self.image_transposition)
        return img, mask

    def _get_batch(self):
        indexes = range(self.index, self.index + self.batch_size)
        existing_indexes = takewhile(lambda i: i < len(self), indexes)
        imgs_masks = tuple(zip(*tuple(map(self.__getitem__, existing_indexes))))
        imgs, masks = tuple(map(np.array, imgs_masks))
        return imgs, masks

    def __iter__(self):
        while self.index < len(self) - self.batch_size:
            yield self._get_batch()
            self.index += self.batch_size


def get_data_generators(normalize=lambda args: args):
    path_tuples = list(utils.read_img_mask_name_pairs(Paths.INPUT_IMGAGES, mask_pattern=r'_L.png$', is_sorted_pairwise=True))
    length = len(path_tuples)
    shuffled = random_permutation(path_tuples)
    dataset_paths = split_into(shuffled, tuple(map(lambda p: int(p*length), Params.dataset_percentages)))
    generators = [CamSeqSequence(paths, batch_size=Params.batch_size, normalize=normalize) for paths in dataset_paths]
    return generators
