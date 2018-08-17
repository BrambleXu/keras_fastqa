import json

from keras import backend as K
from keras.callbacks import Callback

from evaluate_official import evaluate


class FastQALRScheduler(Callback):
    def __init__(self, val_generator, val_answer_file, min_lr=0.0001, steps=1000):
        self.val_generator = val_generator
        with open(val_answer_file) as f:
            val_answers = json.load(f)['data']
        self.val_answers = val_answers
        self.global_steps = 1
        self.steps = steps
        self.last_f1_score = - float('inf')
        self.min_lr = min_lr

    def on_batch_end(self, batch, logs=None):
        if self.global_steps % self.steps == 0:
            predictions = self._predict_answers()
            f1_score = evaluate(self.val_answers, predictions)['f1']
            lr = float(K.get_value(self.model.optimizer.lr))
            if self.last_f1_score > f1_score:
                lr /= 2
                lr = max(lr, self.min_lr)
                K.set_value(self.model.optimizer.lr, lr)
            message = f'\nTraining steps {self.global_steps} (f1_score {f1_score:.2f}): learning rate to {lr:.8f}'
            print(message)
            self.last_f1_score = f1_score
        self.global_steps += 1

    def _predict_answers(self):
        answers = {}
        for inputs, (contexts, ids) in self.val_generator:
            _, _, start_indices, end_indices = self.model.predict_on_batch(inputs)

            for i, (start, end) in enumerate(zip(start_indices, end_indices)):
                answer = ' '.join(contexts[i][j] for j in range(start, end + 1))
                answers[ids[i]] = answer
        self.val_generator.reset()
        return answers


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
