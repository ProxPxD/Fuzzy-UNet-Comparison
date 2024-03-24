import os
from collections import Counter
from typing import Optional, Callable

import tensorflow as tf


class Trainer:
    def __init__(self,
                 model,
                 *,
                 model_name: str = 'model',
                 epochs: int = 5,
                 get_model_path: Callable[[str, int, int], str] = lambda name, epoch, iteration: f'{name}_e{epoch}',
                 save_every_n_epoch: int = 1,
                 save_every_n_iteration: int = None,
                 validate_every_n_epoch: int = None,
                 optimizer = None,
                 metrics = None,
                 loss = None,
                 logger = print,
                 device: str = None,
                 verbose: Optional[int] = 1,
                 batch_size: int = None,
                **kwargs
        ):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.get_model_path = get_model_path
        self.save_every_n_epoch = save_every_n_epoch
        self.save_every_n_iteration = save_every_n_iteration
        self.validate_every_n_epoch = validate_every_n_epoch
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device or ('GPU' if tf.config.experimental.list_physical_devices('GPU') else 'CPU')
        self.batch_size = batch_size
        self.verbose = verbose or 0
        self.logger = logger
        self.counter = Counter()

        self.epoch = None
        self.iteration = None

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def _is_in_right_iteration(self):
        return self.iteration % 1 == 0

    def _verbosely_print(self, required_verbose, to_print, and_cond=lambda: True):
        if self.verbose >= required_verbose and and_cond():
            self.logger(to_print)

    def _count(self, full_label: str, to_write):
        self.counter[full_label] += 1
        # TODO: Add writing to tensorboard

    def train(self, train, test, validation=None, epochs=None, verbose=None):
        self._verbosely_print(1, f'Starting running on {self.device}')
        if epochs:
            self.epochs = epochs
        if verbose is not None:
            self.verbose = verbose
        for epoch in range(self.epochs):
            self.epoch = epoch
            self._verbosely_print(1, f'---Epoch-{epoch+1}/{self.epochs}----------------------')
            for iteration, (X, results) in enumerate(train):
                self.iteration = iteration
                self._verbosely_print(2, f'Iteration {iteration+1:>3}:', self._is_in_right_iteration)
                preds = self.model(X)
                if self.metrics:
                    self._gather_metrics(results, preds)
                print(f'preds shape: {preds.shape}')
                print(f'preds shape[0]: {preds.shape[0]}')
                print(f'results shape: {results.shape}')
                print(f'results shape[0]: {results.shape[0]}')
                if self.batch_size and preds.shape[0] != self.batch_size:
                    self._verbosely_print(2, f'Batch sizes do not match! preds({preds.shape}), results({results.shape})')
                    continue
                self._backwards(results, preds)
                # del X; del preds  # Tensorflow handles memory automatically
                tf.keras.backend.clear_session()
            if self.should_validate() and validation is not None:
                self.validate(validation)

            if self.should_save():
                self.save()

    def _gather_metrics(self, results, preds, mode: str = 'train'):
        for metric_name, metric in self.metrics.items():
            self._verbosely_print(3, f'Calculating {metric_name} for {mode}')
            metric_result = metric(results, preds).numpy()
            full_label = f'{metric_name} - {mode}'
            self._count(full_label, metric_result)

    def _backwards(self, results, preds):
        self._verbosely_print(3, f'Calculating loss')
        loss_result = self.loss(results, preds)
        full_label = f'Loss - train'
        self._count(full_label, loss_result.numpy())
        self._verbosely_print(3, f'{full_label}: {loss_result}')

    def should_save(self):
        return self.get_model_path and self.epoch and self.epoch % self.save_every_n_epoch == 0

    def should_validate(self):
        return self.epoch and self.validate_every_n_epoch is not None and self.epoch % self.validate_every_n_epoch == 0

    def validate(self, validation: tf.data.Dataset):
        self.model.eval()
        self._verbosely_print(2, 'Validating')
        for iteration, (X, results) in enumerate(validation):
            self._verbosely_print(4, f'Iteration {iteration+1:>3}:')
            with tf.device(self.device):
                preds = self.model(X)
                if self.metrics:
                    self._gather_metrics(results, preds, 'val')
        self.model.train()

    def save(self):
        path = self.get_model_path(self.model_name, self.epoch, self.iteration)
        if self.verbose > 2:
            self.logger(f'Saving {path}')
        self.model.save_weights(path)

    def load(self, epoch=0, iteration=0):
        path = self.get_model_path(self.model_name, epoch, iteration)
        path = '../input/img-seg-comp/models/' + path
        last_existing = None
        while os.path.exists(path):
            last_existing = path
            iteration += 1
            path = self.get_model_path(self.model_name, epoch, iteration)
            if path == last_existing or not os.path.exists(path):
                epoch += 1
                iteration = 0
                path = self.get_model_path(self.model_name, epoch, iteration)
                path = '../input/img-seg-comp/models/' + path
        print(path)
        self._verbosely_print(3, f'Last existing model path: {last_existing}')
        self.load_if_exists(last_existing)

    def load_if_exists(self, path):
        existed = os.path.exists(path)
        if existed:
            if self.verbose:
                self.logger(f'Loading {path}')
            self.model.load_weights(path)
        return existed
