from dataclasses import dataclass, asdict
from pathlib import Path

import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy, Precision, Recall, MeanIoU


@dataclass
class Paths:
    DATA = Path(__file__).parent / 'data/data'
    INPUT_DATA = DATA / 'input'
    INPUT_IMGAGES = INPUT_DATA / 'img'
    INPUT_LABELS = INPUT_DATA / 'label_colors.txt'

    OUTPUT_DATA = DATA / 'output'


@dataclass
class DatasetPercentages:
    train: float = .8
    val: float = .1
    test: float = .1


@dataclass
class Params:
    lr = 0e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = CategoricalCrossentropy()

    batch_size = 10
    epochs = 100
    random_split_seed = 42
    log_dir = 'logs'
    dataset_percentages = tuple(asdict(DatasetPercentages()).values())


@dataclass
class Other:  # TODO: think of name
    n_classes = 32
    image_original_size = (720, 960, 3)
    permutated_image_size = (3, 720, 960)

    normalized_image_size = (3, 360, 480)


@dataclass
class Labels:
    COLOR = 'Color'
    CLASS_NAME = 'Class-Name'


L = Labels


def get_metrics(n_classes):
    return {
        'Accuracy': Accuracy(),
        'Precision': Precision(),
        'Recall': Recall(),
        'IoU': MeanIoU(num_classes=n_classes)
    }
