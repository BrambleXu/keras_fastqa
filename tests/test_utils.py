from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch, mock_open, call
from utils import char_span_to_token_span, get_spans, evaluate, filter_dataset, \
    make_small_dataset, split_dataset


class TestUitls(TestCase):
    def test_char_span_to_token_span(self):
        token_offsets = [(0, 4), (5, 6), (7, 11), (12, 14), (15, 16), (17, 21),
                         (21, 22), (23, 26), (27, 31), (32, 37), (38, 47), (47, 48)]
        char_start = 0
        char_end = 11
        span, error = char_span_to_token_span(token_offsets, char_start, char_end)

        self.assertEqual(error, False)
        self.assertEqual(span, (0, 2))

    def test_get_spans(self):
        import spacy
        spacy_en = spacy.load(
            'en_core_web_sm', disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])
        text = 'Rock n Roll is a risk. You risk being ridiculed.'
        contexts = [[token for token in spacy_en(text) if not token.is_space]]
        spans = get_spans(contexts, [0], [11])

        self.assertEqual(spans[0], (0, 2))

    def test_evaluate(self):
        from metrics import SquadMetric
        import numpy as np

        metric = SquadMetric()
        model = Mock()
        start_prob = np.array([[0, 1, 0, 0, 0]], dtype=np.float32)
        end_prob = np.array([[0, 0, 1, 0, 0]], dtype=np.float32)
        model.configure_mock(**{'predict_on_batch.return_value': [start_prob, end_prob]})
        test_generator = MagicMock()
        context = np.array([[1, 2, 3, 4, 5]])
        question = np.array([[6, 7, 8]])
        answer = ['world cup']
        test_generator.__iter__.return_value = iter([[[question, context], answer]])
        index_to_token = {2: 'world', 3: 'cup'}

        em_score, f1_score = evaluate(model, test_generator, metric, index_to_token, 3)
        self.assertEqual(em_score, 1.)
        self.assertEqual(f1_score, 1.)
        model.predict_on_batch.assert_called_with([question, context])

    def test_filter_dataset(self):
        filename = '/path/to/dataset.tsv'
        dest_path = '/path/to/dataset_filtered.tsv'
        read_data = 'Rock n Roll is a risk. You rick being ridiculed.\tDo you like rock music?\n' \
            'Rock n Roll is a risk.\tDo you like rock?'
        open_ = patch('utils.open', mock_open(read_data=read_data)).start()
        open_.return_value.__iter__.return_value = read_data.split('\n')
        writer = patch('csv.writer').start()
        filter_dataset(filename, 5, 7)
        open_.assert_has_calls([call(filename), call(dest_path, 'w')], any_order=True)
        writer.assert_called_once_with(open_.return_value, delimiter='\t')
        writer.return_value.writerow.assert_called_once_with(read_data.split('\n')[1].split('\t'))
        patch.stopall()

    def test_make_small_dataset(self):
        filename = '/path/to/dataset.tsv'
        dest_path = '/path/to/dataset_size_1.tsv'
        read_data = 'Rock n Roll is a risk. You rick being ridiculed.\tDo you like rock music?\n' \
            'Rock n Roll is a risk.\tDo you like rock?'
        open_ = patch('utils.open', mock_open(read_data=read_data)).start()
        open_.return_value.__iter__.return_value = read_data.split('\n')
        writer = patch('csv.writer').start()
        make_small_dataset(filename, 1)
        open_.assert_has_calls([call(filename), call(dest_path, 'w')], any_order=True)
        writer.assert_called_once_with(open_.return_value, delimiter='\t')
        writer.return_value.writerow.assert_called_once()
        patch.stopall()

    def test_split_dataset(self):
        filename = '/path/to/dataset.tsv'
        train_filename = '/path/to/dataset_train.tsv'
        dev_filename = '/path/to/dataset_dev.tsv'
        read_data = 'Rock n Roll is a risk. You rick being ridiculed.\tDo you like rock music?\n' \
            'Rock n Roll is a risk.\tDo you like rock?'
        open_ = patch('utils.open', mock_open(read_data=read_data)).start()
        open_.return_value.__iter__.return_value = read_data.split('\n')
        writer = patch('csv.writer').start()
        split_dataset(filename)
        open_.assert_has_calls(
            [call(filename), call(train_filename, 'w'), call(dev_filename, 'w')],
            any_order=True)
        writer.assert_called_with(open_.return_value, delimiter='\t')
        writer.return_value.writerow.assert_called()
        patch.stopall()
