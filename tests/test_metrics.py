from unittest import TestCase
from metrics import normalize_answer, f1_score, exact_match_score, metric_max_over_ground_truths, SquadMetric


class TestNormalizeAnswer(TestCase):
    def test_normalize_answer(self):
        text = '   Rock    n  Roll.'
        self.assertEqual(normalize_answer(text), 'rock n roll')


class TestF1Score(TestCase):
    def test_f1_score(self):
        prediction = 'rock'
        ground_truth = 'rock n roll'
        precision = 1. * 1 / 1
        recall = 1. * 1 / 3
        f1 = (2 * precision * recall) / (precision + recall)
        self.assertEqual(f1_score(prediction, ground_truth), f1)
        self.assertEqual(f1_score(ground_truth, ground_truth), 1)


class TestExactMatchScore(TestCase):
    def test_exact_match_score(self):
        prediction = 'rock'
        ground_truth = 'rock n roll'
        self.assertEqual(exact_match_score(prediction, ground_truth), 0)
        self.assertEqual(exact_match_score(ground_truth, ground_truth), 1)


class TestMetricMaxOverGroundTruths(TestCase):
    def test_metric_max_over_ground_truths(self):
        prediction = 'rock'
        ground_truth = 'rock n roll'
        precision = 1. * 1 / 1
        recall = 1. * 1 / 3
        f1 = (2 * precision * recall) / (precision + recall)
        self.assertEqual(
            metric_max_over_ground_truths(f1_score, prediction, [ground_truth]), f1)
        self.assertEqual(
            metric_max_over_ground_truths(exact_match_score, prediction, [ground_truth]), 0)
        self.assertEqual(
            metric_max_over_ground_truths(exact_match_score, ground_truth, [ground_truth]), 1)


class TestSquadMetric(TestCase):
    def setUp(self):
        self.metric = SquadMetric()

    def test_call(self):
        prediction = 'rock'
        ground_truth = 'rock n roll'
        self.metric(prediction, ground_truth)
        self.metric(ground_truth, ground_truth)

        precision = 1. * 1 / 1
        recall = 1. * 1 / 3
        f1 = (2 * precision * recall) / (precision + recall)
        self.assertEqual(self.metric._count, 2)
        self.assertEqual(self.metric._total_em, 1)
        self.assertEqual(self.metric._total_f1, (f1 + 1))

    def test_get_metric(self):
        prediction = 'rock'
        ground_truth = 'rock n roll'
        self.metric(prediction, ground_truth)
        self.metric(ground_truth, ground_truth)

        precision = 1. * 1 / 1
        recall = 1. * 1 / 3
        f1 = (2 * precision * recall) / (precision + recall)
        metric = self.metric.get_metric()
        self.assertEqual(metric[0], 1 / 2)
        self.assertEqual(metric[1], (f1 + 1) / 2)
