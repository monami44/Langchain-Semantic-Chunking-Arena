import unittest
from src.evaluation.chunk_size_evaluator import evaluate_chunk_sizes
from src.evaluation.retrieval_quality_evaluator import evaluate_retrieval_quality

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.mock_results = {
            'Percentile': {
                'articles': ["This is a sample chunk."] * 10,
                'news': ["Another sample chunk."] * 15
            },
            'StdDeviation': {
                'articles': ["Sample text for testing."] * 10,
                'news': ["More sample text here."] * 15
            }
        }
        self.ground_truths = {
            'articles': [1]*10,
            'news': [0]*15
        }

    def test_evaluate_chunk_sizes(self):
        metrics = evaluate_chunk_sizes(self.mock_results)
        self.assertIn('Percentile', metrics)
        self.assertIn('StdDeviation', metrics)
        self.assertIn('articles', metrics['Percentile'])
        self.assertIn('news', metrics['StdDeviation'])
        # Check if metrics contain expected keys
        expected_keys = {'mean_size', 'median_size', 'std_dev', 'min_size', 'max_size'}
        for method in metrics:
            for domain in metrics[method]:
                self.assertTrue(expected_keys.issubset(metrics[method][domain].keys()))

    def test_evaluate_retrieval_quality(self):
        metrics = evaluate_retrieval_quality(self.mock_results, self.ground_truths)
        self.assertIn('Percentile', metrics)
        self.assertIn('StdDeviation', metrics)
        self.assertIn('articles', metrics['Percentile'])
        self.assertIn('news', metrics['StdDeviation'])
        # Check if metrics contain expected keys
        expected_keys = {'precision', 'recall', 'f1_score'}
        for method in metrics:
            for domain in metrics[method]:
                self.assertTrue(expected_keys.issubset(metrics[method][domain].keys()))

if __name__ == '__main__':
    unittest.main()
