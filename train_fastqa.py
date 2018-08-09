import os
from argparse import ArgumentParser

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from models import FastQA
from data import SquadReader, Iterator, SquadConverter, Vocabulary
from trainer import SquadTrainer
from callbacks import FastQALRScheduler
from utils import dump_graph

from prepare_vocab import PAD_TOKEN, UNK_TOKEN


def main(args):
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

    root, _ = os.path.splitext(args.vocab_file)
    basepath, basename = os.path.split(root)
    embed_path = f'{basepath}/embedding_{basename}.npy'
    embeddings = np.load(embed_path) if os.path.exists(embed_path) else None

    model = FastQA(len(token_to_index), args.embed, args.hidden,
                   dropout=args.dropout, pretrained_embeddings=embeddings).build()
    opt = Adam()
    model.compile(optimizer=opt, loss_weights=[1, 1, 0, 0],
                  loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', None, None])
    train_dataset = SquadReader(args.train_path)
    dev_dataset = SquadReader(args.dev_path)
    converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN, lower=args.lower)
    train_generator = Iterator(train_dataset, args.batch, converter)
    dev_generator = Iterator(dev_dataset, args.batch, converter)
    trainer = SquadTrainer(model, train_generator, args.epoch, dev_generator,
                           './models/fastqa.{epoch:02d}-{val_loss:.2f}.h5')
    trainer.add_callback(FastQALRScheduler(dev_generator))
    if args.use_tensorboard:
        trainer.add_callback(TensorBoard(log_dir='./graph', batch_size=args.batch))
    history = trainer.run()
    dump_graph(history, 'loss_graph.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--embed', default=300, type=int)
    parser.add_argument('--hidden', default=300, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--train-path', default='./data/train-v1.1_filtered_train.txt', type=str)
    parser.add_argument('--dev-path', default='./data/train-v1.1_filtered_dev.txt', type=str)
    parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
    parser.add_argument('--vocab-file', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--use-tensorboard', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
