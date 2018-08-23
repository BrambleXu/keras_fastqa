import os
from argparse import ArgumentParser

from data import load_squad_tokens, Vocabulary, get_tokenizer


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def main(args):
    tokenizer = get_tokenizer(lower=args.lower, as_str=True)

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

    squad_tokens = load_squad_tokens([args.train_path, args.dev_path],
                                     tokenizer, indices=indices)
    Vocabulary.build(squad_tokens, args.min_freq, args.max_size, (PAD_TOKEN, UNK_TOKEN), filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vocab-path', default='./data/vocab.pkl', type=str)
    parser.add_argument('--train-path', default='./data/train-v1.1.txt', type=str)
    parser.add_argument('--dev-path', default='./data/dev-v1.1.txt', type=str)
    parser.add_argument('--min-freq', default=10, type=int)
    parser.add_argument('--max-size', default=None, type=int)
    parser.add_argument('--only-question', default=False, action='store_true')
    parser.add_argument('--only-context', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
