import math

from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback


class SquadTrainer:
    def __init__(self, model, train_generator, epoch, dev_generator, save_path):
        self.model = model
        self.train_generator = train_generator
        self.dev_generator = dev_generator
        self.epoch = epoch
        self.callbacks = [ModelCheckpoint(save_path)]

    def run(self):
        return self.model.fit_generator(
            generator=self.train_generator, epochs=self.epoch, validation_data=self.dev_generator or None,
            steps_per_epoch=len(self.train_generator), validation_steps=len(self.dev_generator),
            callbacks=self.callbacks)

    def add_callback(self, callback):
        self.callbacks.append(callback)


class BatchLearningRateScheduler(Callback):
    def on_train_begin(self, logs={}):
        self.global_step = 1
        lr = min(0.001, 0.001 / math.log(1000) * math.log(self.global_step))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs={}):
        self.global_step += 1
        if self.global_step <= 1000:
            lr = min(0.001, 0.001 / math.log(1000) * math.log(self.global_step))
            K.set_value(self.model.optimizer.lr, lr)


class ExponentialMovingAverage(Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay

    def on_train_begin(self, logs={}):
        weights = {}
        for weight in self.model.trainable_weights:
            weights[weight.name] = K.get_value(weight)
        self.weights = weights

    def on_batch_end(self, batch, logs={}):
        for weight in self.model.trainable_weights:
            old_weight = self.weights[weight.name]
            self.weights[weight.name] = self.decay * old_weight + (1 - self.decay) * K.get_value(weight)

    def on_epoch_end(self, logs={}):
        for weight in self.model.trainable_weights:
            K.set_value(weight, self.weights[weight.name])
