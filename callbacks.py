from keras import backend as K
from keras.callbacks import Callback


class FastQALRScheduler(Callback):
    def __init__(self, val_generator, steps=1000):
        self.val_generator = val_generator
        self.val_steps = len(val_generator)
        self.global_steps = 1
        self.steps = 1000
        self.last_val_loss = float('inf')

    def on_batch_end(self, batch, logs=None):
        if self.global_steps % self.steps == 0:
            val_loss = self.model.evaluate_generator(
                self.val_generator, steps=self.val_steps, workers=0)[0]
            if self.last_val_loss < val_loss:
                lr = float(K.get_value(self.model.optimizer.lr))
                lr /= 2
                K.set_value(self.model.optimizer.lr, lr)
                message = f'\nTraining steps {self.global_steps}: LearningRateScheduler setting learning rate to {lr}'
                print(message)
            self.last_val_loss = val_loss
        self.global_steps += 1
