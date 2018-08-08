from argparse import ArgumentParser

from models import FastQA
from data import SquadReader, Iterator, Vocabulary, SquadTestConverter
from metrics import SquadMetric
from utils import evaluate

from prepare_vocab import PAD_TOKEN, UNK_TOKEN


def main(args):
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

    model = FastQA(len(token_to_index), args.embed, args.hidden).build()
    model.load_weights(args.model_path)

    metric = SquadMetric()
    test_dataset = SquadReader(args.test_path)
    converter = SquadTestConverter(token_to_index, PAD_TOKEN, UNK_TOKEN, lower=args.lower)
    test_generator = Iterator(test_dataset, args.batch, converter, False, False)
    em_score, f1_score = evaluate(model, test_generator, metric, index_to_token)
    print('EM: {}, F1: {}'.format(em_score, f1_score))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--embed', default=300, type=int)
    parser.add_argument('--hidden', default=300, type=int)
    parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
    parser.add_argument('--vocab-file', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--model-path', type=str)
    args = parser.parse_args()
    main(args)
