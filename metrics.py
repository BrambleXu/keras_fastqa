import string
from collections import Counter


def normalize_answer(text):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(char for char in text if char not in exclude)

    return white_space_fix(remove_punc(str.lower(text)))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth == '':
        return 1
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1. * num_same / len(prediction_tokens)
    recall = 1. * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_groud_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_groud_truths.append(score)
    return max(scores_for_groud_truths)


class SquadMetric:
    def __init__(self):
        self._total_em = 0.
        self._total_f1 = 0.
        self._count = 0

    def __call__(self, best_span_string, answer_string):
        em = metric_max_over_ground_truths(
            exact_match_score, best_span_string, [answer_string])
        f1 = metric_max_over_ground_truths(
            f1_score, best_span_string, [answer_string])
        self._total_em += em
        self._total_f1 += f1
        self._count += 1

    def get_metric(self, reset=False):
        em = self._total_em / self._count if self._count > 0 else 0
        f1 = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self._total_em = 0.
            self._total_f1 = 0.
            self._count = 0
        return em, f1
