from unittest.mock import patch, mock_open
from unittest import TestCase

import numpy as np
from data import make_vocab, load_squad_tokens, SquadReader, Iterator,\
    SquadConverter, SquadTestConverter, Vocabulary


class TestData(TestCase):
    def test_make_vocab(self):
        tokens = ['Rock', 'n', 'Roll', 'is', 'a', 'risk', '.', 'You', 'rick',
                  'being', 'ridiculed', '.']
        token_to_index, index_to_token = make_vocab(tokens, 1, 10)

        self.assertEqual(token_to_index['<pad>'], 0)
        self.assertEqual(token_to_index['<unk>'], 1)
        self.assertEqual(token_to_index['<s>'], 2)
        self.assertEqual(token_to_index['</s>'], 3)
        self.assertEqual(len(token_to_index), 10)
        self.assertEqual(len(index_to_token), 10)
        self.assertEqual(index_to_token[0], '<pad>')
        self.assertEqual(index_to_token[1], '<unk>')
        self.assertEqual(index_to_token[2], '<s>')
        self.assertEqual(index_to_token[3], '</s>')

    def test_load_squad_tokens(self):
        tokenizer = str.split
        read_data = 'a b c d\te f g h i j k\nl m n o p\tq r s t u'
        filename = '/path/to/squad.tsv'
        open_ = patch('data.open', mock_open(read_data=read_data)).start()
        open_.return_value.__iter__.return_value = read_data.split('\n')
        tokens = load_squad_tokens(filename, tokenizer)
        self.assertCountEqual(list(tokens), read_data.split())
        open_.assert_called_with(filename)
        patch.stopall()


class TestVocabulary(TestCase):
    def test_build(self):
        tokens = ['rock', 'n', 'roll']
        token_to_index, index_to_token = Vocabulary.build(tokens, 1, 4, ('<pad>',), None)
        tokens += ['<pad>']
        self.assertCountEqual(token_to_index.keys(), tokens)

    def test_load(self):
        filename = '/path/to/vocab.pkl'
        open_ = patch('data.open', mock_open()).start()
        pickle_load = patch('data.pickle.load').start()
        pickle_load.return_value = ('token_to_index', 'index_to_token')
        token_to_index, index_to_token = Vocabulary.load(filename)
        self.assertEqual(token_to_index, 'token_to_index')
        self.assertEqual(index_to_token, 'index_to_token')
        open_.assert_called_with(filename, mode='rb')
        pickle_load.assert_called_with(open_.return_value)


class TestSquadReader(TestCase):
    def setUp(self):
        read_data = 'context1\tquestion1\tstart1\tend1\tanswer1\n' \
            'context2\tquestion2\tstart2\tend2\tanswer2'
        self.filename = 'path/to/target.tsv'
        self.lines = read_data.split('\n')
        self.mock_open = patch('data.open', mock_open(read_data=read_data)).start()
        self.mock_getline = patch('data.linecache.getline').start()
        self.dataset = SquadReader(self.filename)

    def test_init(self):
        self.assertEqual(self.dataset._filename, self.filename)
        self.assertEqual(self.dataset._total_data, 2)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        for i, line in enumerate(self.lines):
            self.mock_getline.return_value = line
            self.assertListEqual(self.dataset[i], line.split('\t'))
            self.mock_getline.assert_called_with(self.filename, i + 1)

    def tearDown(self):
        patch.stopall()


class TestIterator(TestCase):
    def setUp(self):
        dataset = range(100)
        self.batch_size = 32

        def converter(x): return x

        self.generator1 = Iterator(dataset, self.batch_size, converter)
        self.generator2 = Iterator(dataset, self.batch_size, converter, False, False)
        self.dataset = dataset
        self.converter = converter

    def test_init(self):
        self.assertEqual(self.generator1._dataset, self.dataset)
        self.assertEqual(self.generator1._batch_size, self.batch_size)
        self.assertEqual(self.generator1._converter, self.converter)
        self.assertEqual(self.generator1._current_position, 0)
        self.assertEqual(len(self.generator1._order), len(self.generator1._dataset))

        self.assertEqual(self.generator2._dataset, self.dataset)
        self.assertEqual(self.generator2._batch_size, self.batch_size)
        self.assertEqual(self.generator2._converter, self.converter)
        self.assertEqual(self.generator2._current_position, 0)
        self.assertEqual(self.generator2._order, None)

    def test_len(self):
        self.assertEqual(len(self.generator1), 4)
        self.assertEqual(len(self.generator2), 4)

    def test_next(self):
        for i in range(10):
            batch = next(self.generator1)
            self.assertEqual(len(batch), self.batch_size)
        for i, batch in enumerate(self.generator2):
            if i < len(self.generator2) - 1:
                self.assertEqual(len(batch), self.batch_size)
            else:
                self.assertEqual(len(batch), 4)

    def test_reset(self):
        self.generator1.reset()
        self.assertEqual(self.generator1._current_position, 0)
        self.assertEqual(len(self.generator1._order), len(self.generator1._dataset))

        self.generator2.reset()
        self.assertEqual(self.generator2._current_position, 0)
        self.assertEqual(self.generator2._order, None)

    def test_iter(self):
        self.assertEqual(self.generator1.__iter__(), self.generator1)
        self.assertEqual(self.generator2.__iter__(), self.generator2)


class TestSquadConverter(TestCase):
    batch = [[
        'Rock n Roll is a risk. You risk being ridiculed.',
        'What is your risk?',
        38, 47,
        'ridiculed'
    ]]

    token_to_index = {'<pad>': 0, '<unk>': 1, 'is': 2, 'a': 3, 'risk': 4,
                      '.': 5, 'you': 6, 'being': 7, 'ridiculed': 8, 'what': 9,
                      'your': 10}

    def setUp(self):
        self.converter = SquadConverter(
            self.token_to_index, '<pad>', '<unk>', True, 5, 12)

    def test_init(self):
        self.assertEqual(self.converter._unk_index, 1)
        self.assertEqual(self.converter._pad_token, '<pad>')
        self.assertEqual(self.converter._token_to_index, self.token_to_index)

    def test_call(self):
        inputs, outputs = self.converter(self.batch)
        question = np.array([[9, 2, 10, 4, 1]], dtype=np.int32)
        context = np.array([[1, 1, 1, 2, 3, 4, 5, 6, 4, 7, 8, 5]], dtype=np.int32)
        start_index = np.array([10], dtype=np.int32)
        end_index = np.array([10], dtype=np.int32)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 2)
        np.testing.assert_array_equal(inputs[0], question)
        np.testing.assert_array_equal(inputs[1], context)
        np.testing.assert_array_equal(outputs[0], start_index)
        np.testing.assert_array_equal(outputs[1], end_index)

    def test_process_text(self):
        contexts = [self.converter._tokenizer(self.batch[0][0])]
        batch = self.converter._process_text(contexts, 12)
        context = np.array([[1, 1, 1, 2, 3, 4, 5, 6, 4, 7, 8, 5]], dtype=np.int32)
        np.testing.assert_array_equal(batch, context)


class TestSquadTestConverter(TestSquadConverter):
    def setUp(self):
        self.converter = SquadTestConverter(
            self.token_to_index, '<pad>', '<unk>', True, 5, 12)

    def test_call(self):
        inputs, output = self.converter(self.batch)
        question = np.array([[9, 2, 10, 4, 1]], dtype=np.int32)
        context = np.array([[1, 1, 1, 2, 3, 4, 5, 6, 4, 7, 8, 5]], dtype=np.int32)
        self.assertEqual(len(inputs), 2)
        np.testing.assert_array_equal(inputs[0], question)
        np.testing.assert_array_equal(inputs[1], context)
        self.assertEqual(output, ['ridiculed'])

    def test_get_valid_tokenized_answers(self):
        answer = 'well-done'
        valid_answer = 'well - done'
        result = self.converter._get_valid_tokenized_answers([answer])
        self.assertListEqual(result, [valid_answer])
