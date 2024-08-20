import re
import numpy as np
import cv2
from itertools import product
from pathlib import Path
from typing import Callable, Iterable
from pydash import spread

from constants import Params, L, Other

path_like = Path | str


def read_img_mask_name_pairs(
        img_path: path_like,
        mask_path: path_like = None,
        img_pattern=None,
        mask_pattern=None,
        is_sorted_pairwise=True,
        img_and_mask_match_cond: Callable[[path_like, path_like], bool] = None,
) -> Iterable[tuple[Path, Path]]:
    mask_path = mask_path or img_path
    img_path = Path(mask_path) if mask_path else img_path
    compiled_mask_pattern = re.compile(mask_pattern or r'.png$')
    mask_cond = lambda mask: compiled_mask_pattern.search(str(mask))
    if img_path == mask_path and img_pattern is None and mask_pattern is not None:
        img_cond = lambda img: not mask_cond(img)
    else:
        compiled_img_pattern = re.compile(img_pattern or r'.png$')
        img_cond = lambda img: compiled_img_pattern.search(str(img))
    imgs = filter(img_cond, img_path.iterdir())
    masks = filter(mask_cond, mask_path.iterdir())
    if is_sorted_pairwise:
        return zip(sorted(imgs), sorted(masks))
    if img_and_mask_match_cond is None:
        img_and_mask_match_cond = lambda img, mask: img.stem in mask.stem
    return filter(img_and_mask_match_cond, product(imgs, masks))


def path_to_numpy(iterable: Iterable, normalize: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda *args: args) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for img_path, mask_path in iterable:
        load = lambda path: cv2.imread(str(path))
        img, mask = load(img_path), load(mask_path)
        img, mask = normalize(img, mask)
        yield img, mask


def get_resize(shape):
    def resize(img):
        img = img.transpose((2, 0, 1))
        img = cv2.resize(img, shape[::-1])
        img.transpose((1, 2, 0))
        return img
    return resize


def normalize_mask(mask, resize: Callable = lambda img: img):
    mask = resize(mask)
    return mask


def normalize_picture(img: np.ndarray, resize: Callable = None):
    resize = resize or (lambda img: img)
    img = img.astype(np.float32) / 255
    img = resize(img)
    return img


def get_normalize(labels):
    return spread(lambda X, mask: (normalize_picture(X), normalize_mask(mask)))
