import os
from argparse import ArgumentParser

import spacy

from data import load_squad_tokens, Vocabulary


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def main(args):
    spacy_en = spacy.load('en_core_web_sm', disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    postprocess = str.lower if args.lower else lambda x: x

    def tokenizer(x):
        return [postprocess(token.text) for token in spacy_en(x) if not token.is_space]

    if args.only_question:
        indices = [1]
        desc = 'question'
    elif args.only_context:
        indices = [0]
        desc = 'context'
    else:
        indices = [0, 1]
        desc = 'question_context'

    basename, ext = os.path.splitext(args.vocab_path)
    min_freq = args.min_freq if args.min_freq else ''
    max_size = args.max_size if args.max_size else ''
    filename = f'{basename}_{desc}_min-freq{min_freq}_max_size{max_size}{ext}'

    squad_tokens = load_squad_tokens(args.train_path, tokenizer, indices=indices)
    Vocabulary.build(squad_tokens, args.min_freq, args.max_size, (PAD_TOKEN, UNK_TOKEN), filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vocab-path', default='./data/vocab.pkl', type=str)
    parser.add_argument('--train-path', default='./data/train-v1.1_filtered_train.txt', type=str)
    parser.add_argument('--min-freq', default=10, type=int)
    parser.add_argument('--max-size', default=None, type=int)
    parser.add_argument('--only-question', default=False, action='store_true')
    parser.add_argument('--only-context', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
