# FastQA by Keras

FastQA implemented by Keras

## Description

This repository is implementation of FastQA proposed in [this paper](https://arxiv.org/abs/1703.04816)

## Requirement

* Python 3.6+
* TensorFlow, Keras, NumPy, spaCy

## Usage

Building vocabulary

```py
import os
import spacy

from data import Vocabulary, load_squad_tokens

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if __name__ == '__main__':
    train_file = '/path/to/train.txt'
    vocab_file = 'vocab.pkl'
    squad_tokens = load_squad_tokens(train_file)
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, min_freq, max_size, (PAD_TOKEN, UNK_TOKEN), vocab_file)
```

Training model

```py
from models import FastQA
from data import SquadReader, Iterator, SquadConverter
from trainer import SquadTrainer


model = FastQA(vocab_size, embed_size, hidden_size).build()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
dataset = SquadReader(train_file)
converter = SquadConverter(token_to_index, '<pad>', '<unk>')
train_generator = Iterator(dataset, batch_size, converter)
trainer = SquadTrainer(model, train_generator, epoch)
trainer.run()
```

## Install

```sh
$ git clone https://github.com/yasufumy/keras_fastqa.git
```
