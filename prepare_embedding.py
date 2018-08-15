import os
import pickle
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from data import Vocabulary
from prepare_vocab import PAD_TOKEN, UNK_TOKEN


def save_word_embedding_as_npy(filename, dim):
    npy_name = f'{filename}.npy'
    dict_name = f'{filename}.dict'
    embeddings = []
    token_to_index = {}
    with open(filename) as f:
        for line in tqdm(f):
            elements = line.split()
            word = ''.join(elements[0:-dim])
            vector = [float(x) for x in elements[-dim:]]
            if len(vector) != dim:
                continue
            embeddings.append(vector)
            token_to_index[word] = len(token_to_index)
    with open(dict_name, 'wb') as f:
        pickle.dump(token_to_index, f)

    embeddings = np.array(embeddings, dtype=np.float32)

    np.save(npy_name, embeddings)

    return token_to_index, embeddings


def extract_embeddings(vocab, pretrained_vocab, pretrained_embeddings, dim=300):
    embeddings = np.random.randn(len(vocab), 300).astype(np.float32)
    for word, index in vocab.items():
        if word in pretrained_vocab:
            vector = pretrained_embeddings[pretrained_vocab[word]]
            embeddings[index] = vector
    if PAD_TOKEN in vocab:
        embeddings[vocab[PAD_TOKEN]] = 0.
    if UNK_TOKEN in vocab:
        embeddings[vocab[UNK_TOKEN]] = pretrained_embeddings[pretrained_vocab['unk']]
    return embeddings


def main(args):
    token_to_index, _ = Vocabulary.load(args.vocab_path)

    if os.path.exists(args.embed_array_path) and os.path.exists(args.embed_dict_path):
        with open(args.embed_dict_path, 'rb') as f:
            pretrained_token_to_index = pickle.load(f)
        embeddings = extract_embeddings(token_to_index, pretrained_token_to_index,
                                        np.load(args.embed_array_path))
    else:
        if os.path.exists(args.embed_path):
            pretrained_token_to_index, embeddings = save_word_embedding_as_npy(args.embed_path, args.dim)
        else:
            raise FileNotFoundError('Please download pre-trained embedding file')
    root, _ = os.path.splitext(args.vocab_path)
    basepath, basename = os.path.split(root)
    filename = f'{basepath}/embedding_{basename}.npy'
    np.save(filename, embeddings)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vocab-path', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--embed-path', default='./data/wiki.en.vec', type=str)
    parser.add_argument('--dim', default=300, type=int)
    parser.add_argument('--embed-array-path', default='./data/wiki.en.vec.npy', type=str)
    parser.add_argument('--embed-dict-path', default='./data/wiki.en.vec.dict', type=str)
    args = parser.parse_args()

    main(args)
