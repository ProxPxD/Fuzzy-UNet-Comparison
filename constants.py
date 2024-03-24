from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy, Precision, Recall, MeanIoU


@dataclass
class Paths:
    DATA = Path('../input/img-seg-comp/data/data')
    INPUT_DATA = DATA / 'input'
    INPUT_IMGAGES = INPUT_DATA / 'img'
    INPUT_LABELS = INPUT_DATA / 'label_colors.txt'

    OUTPUT_DATA = DATA / 'output'


@dataclass
class Params:
    lr = 0e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = CategoricalCrossentropy()

    batch_size = 10
    epochs = 100
    random_split_seed = 42
    log_dir = 'logs'


def get_metrics(n_classes):
    return {
        'Accuracy': Accuracy(),
        'Precision': Precision(),
        'Recall': Recall(),
        'IoU': MeanIoU(num_classes=n_classes)
    }
