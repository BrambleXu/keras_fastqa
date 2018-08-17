import os
import csv
import random
import pickle
import linecache
import json

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa


def char_span_to_token_span(token_offsets, char_start, char_end):
    if char_start < 0:
        return (-1, -1), False

    error = False

    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < char_start:
        start_index += 1
    if token_offsets[start_index][0] > char_start:
        start_index -= 1
    if token_offsets[start_index][0] != char_start:
        error = True

    end_index = start_index
    while end_index < len(token_offsets) and token_offsets[end_index][1] < char_end:
        end_index += 1
    if token_offsets[end_index][1] != char_end:
        error = True
    return (start_index, end_index), error


def get_spans(contexts, starts, ends):
    spans = []
    for context, start, end in zip(contexts, starts, ends):
        context_offsets = [(token.idx, token.idx + len(token.text)) for token in context]
        span, error = char_span_to_token_span(context_offsets, start, end)
        if error:
            ...

        spans.append(span)

    return spans


def dump_graph(history, filename):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('cross entropy loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(filename)


def filter_dataset(filename, question_max_length=50, context_max_length=400):
    import spacy
    import csv
    import os
    from tqdm import tqdm

    spacy_en = spacy.load('en_core_web_sm',
                          disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    def tokenizer(x): return [token for token in spacy_en(x) if not token.is_space]

    with open(filename) as f:
        dataset = [row for row in csv.reader(f, delimiter='\t')]
    basename, ext = os.path.splitext(filename)
    filename = f'{basename}_filtered{ext}'
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for data in tqdm(dataset):
            context_tokens = tokenizer(data[0])
            question_tokens = tokenizer(data[1])
            if len(context_tokens) <= context_max_length and \
               len(question_tokens) <= question_max_length:
                writer.writerow(data)


def make_small_dataset(filename, size=100, overwrite=False):
    basename, ext = os.path.splitext(filename)
    new_filename = f'{basename}_size_{size}{ext}'

    if os.path.exists(new_filename) and not overwrite:
        raise FileExistsError('Target file already exists, set overwrite as True')

    with open(filename) as f:
        num_lines = len(f.readlines())

    indices = random.sample(range(num_lines), size)
    with open(new_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in tqdm(indices):
            line = linecache.getline(filename, i + 1)
            data = next(csv.reader([line], delimiter='\t'))
            writer.writerow(data)


def make_answer_dataset(tsv_file, json_file):
    with open(json_file) as f:
        original_dataset = json.load(f)

    with open(tsv_file) as f:
        preprocessed_dataset_ids = [line[-1] for line in csv.reader(f, delimiter='\t')]

    qas = []
    for article in original_dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] in preprocessed_dataset_ids:
                    qas.append(qa)
    answer_dataset = {'data': [{'paragraphs': [{'qas': qas}]}],
                      'version': original_dataset['version']}

    basename, ext = os.path.splitext(tsv_file)
    answer_dataset_file = f'{basename}.json'
    with open(answer_dataset_file, 'w') as f:
        json.dump(answer_dataset, f, indent=2)


def split_dataset(filename, ratio=0.8, overwrite=False):
    basename, ext = os.path.splitext(filename)
    train_filename = f'{basename}_train{ext}'
    dev_filename = f'{basename}_dev{ext}'

    if os.path.exists(train_filename) and not overwrite:
        raise FileExistsError('Target file already exists, set overwrite as True')

    with open(filename) as f:
        num_lines = len(f.readlines())

    train_size = round(num_lines * ratio)
    indices = list(range(num_lines))
    random.shuffle(indices)

    def write_csv(filename, index, writer):
        line = linecache.getline(filename, index)
        data = next(csv.reader([line], delimiter='\t'))
        writer.writerow(data)

    with open(train_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in tqdm(indices[:train_size]):
            write_csv(filename, i + 1, writer)

    with open(dev_filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in tqdm(indices[train_size:]):
            write_csv(filename, i + 1, writer)


if __name__ == '__main__':
    from allennlp.data.dataset_readers import SquadReader

    import spacy

    spacy_en = spacy.load('en_core_web_sm',
                          disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    def tokenizer(x): return [(token.idx, token.text) for token in spacy_en(x) if not token.is_space]

    if not os.path.exists('data_list.pkl'):
        with open('data/train-v2.0.txt') as f:
            data = [row for row in csv.reader(f, delimiter='\t')]
        data = [[tokenizer(x[0]), int(x[2]), int(x[3]), x[4]]
                for x in data]
        contexts, char_starts, char_ends, answers = zip(*data)
        with open('data_list.pkl', 'wb') as f:
            pickle.dump((contexts, char_starts, char_ends, answers), f)
    else:
        with open('data_list.pkl', 'rb') as f:
            contexts, char_starts, char_ends, answers = pickle.load(f)

    spans = get_spans(contexts, char_starts, char_ends, answers)

    data = SquadReader().read('/Users/smap11/Desktop/train-v2.0.json')

    for span, x in zip(spans, data):
        if 'span_start' in x.fields:
            if span[0] != x.fields['span_start'].sequence_index:
                print('start')
                print(span[0], x.fields['span_start'])
            if span[1] != x.fields['span_end'].sequence_index:
                print('end')
                print(span[1], x.fields['span_end'])
