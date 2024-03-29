import cv2
import numpy as np
from keras.utils import Sequence
from toolz import compose_left

import utils
from constants import Params, Paths


class CamSeqSequence(Sequence):
    def __init__(self, path_tuples, normalize=lambda args: args):
        self.path_tuples = path_tuples
        self.normalize = normalize
        self.image_transposition = (2, 0, 1)
        self.index = 0

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, index):
        load = compose_left(str, cv2.imread, self.normalize)
        img, mask = tuple(map(load, self.path_tuples[index]))
        img = img.transpose(self.image_transposition)
        return img, mask

    def __iter__(self):
        while self.index < len(self.path_tuples):
            yield self[self.index]
            self.index += 1


def get_data_generators(normalize=lambda args: args):
    path_tuples = list(utils.read_img_mask_name_pairs(Paths.INPUT_IMGAGES, mask_pattern=r'_L.png$', is_sorted_pairwise=True))
    dataset_paths = np.array_split(path_tuples, Params.dataset_percentages)
    generators = [CamSeqSequence(paths, normalize) for paths in dataset_paths]
    return generators
