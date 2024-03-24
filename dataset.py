import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical

from constants import Params


class CamSeqSequence(Sequence):
    def __init__(self, path_tuples, normalize=lambda args: args):
        self.path_tuples = path_tuples
        self.normalize = normalize
        self.image_transposition = (2, 0, 1)
        self.index = -1

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, index):
        raise NotImplementedError
        img_path, mask_path = self.path_tuples[index]
        img = utils.read_img(img_path)
        mask = utils.read_img(mask_path)
        img, mask = self.normalize(img), self.normalize(mask)
        img = img.transpose(self.image_transposition)
        return img, mask


def get_data_generators(normalize=lambda args: args):
    path_tuples = list(utils.read_img_mask_name_pairs(Paths.INPUT_IMGAGES, mask_pattern=r'_L.png$', is_sorted_pairwise=True))
    dataset_paths = np.array_split(path_tuples, Parameters.dataset_percentages)
    generators = [CamSeqSequence(paths, normalize) for paths in dataset_paths]
    return generators
