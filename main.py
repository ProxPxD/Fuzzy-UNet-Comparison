from tensorflow.python.training.summary_io import SummaryWriter

import analysis
import dataset
import utils
from constants import Params
from models.models import space, ModelFactory
from trainer_tf import Trainer


labels = analysis.load_labels()
normalize = utils.get_normalize(labels)
train_loader, val_loader, test_loader = dataset.get_dataloaders(normalize)
writer = SummaryWriter(Params.log_dir)
to_analyze = ModelFactory.get_models_to_analyze(space)
trainer = Trainer(
    writer=writer,
    get_model_path=lambda model_name, epoch, iteration: f'models/{model_name}_e{epoch}',
    verbose=3,
    optimizer=Params.optimizer,
    loss=Params.loss,
    validate_every_n_epoch=1,
    epochs=Params.epochs,
)


for name, model in to_analyze.items():
    trainer.set_model(model, name)
    trainer.train(train_loader, val_loader)