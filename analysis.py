
import pandas as pd

import numpy as np
from more_itertools import unique_everseen, flatten

import utils
from constants import Paths, L


def normalize(X, mask):
    return X.astype(np.float32) / 255, mask


def get_img_mask_pairs():
    img_mask_paths_pairs = utils.read_img_mask_name_pairs(Paths.INPUT_IMGAGES, mask_pattern=r'_L.png$', is_sorted_pairwise=True)
    img_mask_pairs = utils.path_to_numpy(img_mask_paths_pairs, normalize)
    return img_mask_pairs


def get_first_pair():
    img_mask_pairs = get_img_mask_pairs()
    pair = next(img_mask_pairs)
    return pair

# Analysis functions


def get_img_format():
    img, mask = get_first_pair()
    return img[0, 0, :]


def get_shape():
    img_mask_pairs = get_img_mask_pairs()
    unique_shapes = unique_everseen(map(lambda elem: elem.shape, flatten(img_mask_pairs)))
    return list(unique_shapes)


def get_maks_format():
    img, mask = get_first_pair()
    mask_pixel = mask[0, 0]
    pixel_type = type(mask_pixel)
    return mask_pixel, mask, pixel_type


def load_labels():
    labels = pd.read_csv(Paths.INPUT_LABELS, sep='\t', header=None)
    labels.columns = [L.COLOR, L.CLASS_NAME]
    labels[L.COLOR] = labels[L.COLOR].apply(lambda text: tuple(map(int, text.split(' ')[::-1])))
    return labels


def map_mask():
    mask_pixel, mask, pixel_type = get_maks_format()
    labels = load_labels()

    label_dict = {tuple(row[L.COLOR]): idx for idx, row in labels.iterrows()}
    flat_mask = mask.reshape(-1, mask.shape[-1])
    label_indices = np.array([label_dict[tuple(pixel)] for pixel in flat_mask], dtype=np.int32)

    # return np.fromiter((labels.index[labels[L.COLOR] == tuple(pixel)][0] for row in mask for pixel in row), dtype=np.int32)
    return label_indices


def get_label_info():
    labels = load_labels()
    n_classes = labels.shape[0]
    return n_classes, labels


if __name__ == '__main__':
    analysis_funcs = {
        'Img format': get_img_format,
        'Unique shapes': get_shape,
        'Mask format': get_maks_format,
        'Labels': get_label_info,
        'Map mask to int': map_mask,
    }

    for a_name, a_func in analysis_funcs.items():
        print(f'{a_name}: ', a_func())

