from torch.utils.tensorboard import FileWriter

import analysis
import dataset
import utils
from constants import Params
from models.models import space, ModelFactory
from trainer_tf import Trainer
import tensorflow as tf


print('GPU devices:', tf.config.list_physical_devices('GPU'))
if is_gpu := tf.test.is_gpu_available():
    print('is GPU')
else:
    print('No GPU')

labels = analysis.load_labels()
normalize = utils.get_normalize(labels)
train_loader, val_loader, test_loader = dataset.get_data_generators(normalize)
writer = FileWriter(Params.log_dir)
to_analyzes = ModelFactory.get_models_to_analyze(space)
trainer = Trainer(
    writer=writer,
    get_model_path=lambda model_name, epoch, iteration: f'models/{model_name}_e{epoch}',
    verbose=3,
    optimizer=Params.optimizer,
    loss=Params.loss,
    validate_every_n_epoch=1,
    epochs=Params.epochs,
)


for name, model in to_analyzes.items():
    trainer.set_model(model, name)
    trainer.train(train_loader, val_loader)
