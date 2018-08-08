from keras import backend as K
from keras.callbacks import Callback


class LearningRateScheduler(Callback):
    def __init__(self):
        self.global_steps = 1

    def on_batch_begin(self, batch, logs=None):
        if self.global_steps % 1000 == 0:
            lr = float(K.get_value(self.model.optimizer.lr))
            lr /= 2
            K.set_value(self.model.optimizer.lr, lr)
            message = f'\nTraining steps {self.global_steps}: LearningRateScheduler setting learning rate to {lr}'
            print(message)
        self.global_steps += 1
