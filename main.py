import logging

from torch.utils.tensorboard import FileWriter

import analysis
import dataset
import utils
from constants import Params, Other
from custom_models.models import space, ModelFactory
from trainer_tf import Trainer
import tensorflow as tf


print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('is GPU' if (is_gpu := tf.test.is_gpu_available()) else 'No GPU')

labels = analysis.load_labels()
normalize = utils.get_normalize(labels)
train_loader, val_loader, test_loader = dataset.get_data_generators(normalize)
writer = FileWriter(Params.log_dir)

to_analyzes = ModelFactory.get_models_to_analyze(
    space,
    n_classes=Other.n_classes,
    all_n_channels=(64, 64),
    input_shape=Other.image_original_size,
)

trainer = Trainer(
    writer=writer,
    get_model_path=lambda model_name, epoch, iteration: f'models/{model_name}_e{epoch}',
    verbose=3,
    optimizer=Params.optimizer,
    loss=Params.loss,
    validate_every_n_epoch=1,
    epochs=Params.epochs,
    batch_size=Params.batch_size,
    logger=logging.debug,
)

for name, model in to_analyzes.items():
    print(f'Setting {name}')
    trainer.set_model(model, name)
    print(f'Training {name}')
    trainer.train(train_loader, val_loader)
