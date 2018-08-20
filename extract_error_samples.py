import sys
import json
from argparse import ArgumentParser

from evaluate_official import exact_match_score, metric_max_over_ground_truths


def extract(dataset, predictions):
    error_samples = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            error_qas = []
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    continue
                ground_truths = [x['text'] for x in qa['answers']]
                prediction = predictions[qa['id']]
                exact_match = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                if not exact_match:
                    qa['wrong_answer'] = prediction
                    error_qas.append(qa)
            paragraph['qas'] = error_qas
            error_samples.append(paragraph)
    return error_samples


if __name__ == '__main__':
    expected_version = '1.1'
    parser = ArgumentParser()
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(extract(dataset, predictions), indent=2))
