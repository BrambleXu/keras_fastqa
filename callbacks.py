from keras import backend as K
from keras.callbacks import Callback


class FastQALRScheduler(Callback):
    def __init__(self, val_generator, steps=1000):
        self.val_generator = val_generator
        self.val_steps = len(val_generator)
        self.global_steps = 1
        self.steps = steps
        self.last_val_loss = float('inf')

    def on_batch_end(self, batch, logs=None):
        if self.global_steps % self.steps == 0:
            val_loss = self.model.evaluate_generator(
                self.val_generator, steps=self.val_steps, workers=0)[0]
            lr = float(K.get_value(self.model.optimizer.lr))
            if self.last_val_loss < val_loss:
                lr /= 2
                K.set_value(self.model.optimizer.lr, lr)
            message = f'\nTraining steps {self.global_steps} (val_loss {val_loss:.2f}): learning rate to {lr:.8f}'
            print(message)
            self.last_val_loss = val_loss
        self.global_steps += 1


class FastQACheckpoint(Callback):
    def __init__(self, filepath, steps=1000):
        super().__init__()
        self.filepath = filepath
        self.global_steps = 1
        self.steps = steps

    def on_batch_end(self, batch, logs=None):
        if self.global_steps % self.steps == 0:
            filepath = self.filepath.format(steps=self.global_steps)
            self.model.save_weights(filepath, overwrite=True)
        self.global_steps += 1
