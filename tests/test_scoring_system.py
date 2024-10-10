import unittest
from src.evaluation.scoring_system import calculate_scores

class TestScoringSystem(unittest.TestCase):
    def setUp(self):
        self.chunk_size_metrics = {
            'Percentile': {
                'articles': {'mean_size': 100, 'std_dev': 10, 'min_size': 90, 'max_size': 110},
                'news': {'mean_size': 120, 'std_dev': 15, 'min_size': 100, 'max_size': 140}
            },
            'StdDeviation': {
                'articles': {'mean_size': 95, 'std_dev': 8, 'min_size': 85, 'max_size': 105},
                'news': {'mean_size': 115, 'std_dev': 12, 'min_size': 95, 'max_size': 135}
            }
        }
        self.retrieval_metrics = {
            'Percentile': {
                'articles': {'precision': 0.8, 'recall': 0.75, 'f1_score': 0.77},
                'news': {'precision': 0.85, 'recall': 0.8, 'f1_score': 0.82}
            },
            'StdDeviation': {
                'articles': {'precision': 0.82, 'recall': 0.78, 'f1_score': 0.80},
                'news': {'precision': 0.88, 'recall': 0.83, 'f1_score': 0.85}
            }
        }

    def test_calculate_scores(self):
        scores = calculate_scores(self.chunk_size_metrics, self.retrieval_metrics)
        
        # Test presence of methods and domains
        self.assertIn('Percentile', scores)
        self.assertIn('StdDeviation', scores)
        self.assertIn('articles', scores['Percentile'])
        self.assertIn('news', scores['StdDeviation'])
        
        # Test score ranges
        for method, domains in scores.items():
            for domain, score in domains.items():
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)
        
        # Test specific score calculations
        # Example for Percentile -> articles
        # Size Score:
        # mean_score = 100 - 100 = 0
        # std_dev_score = 100 - 10 = 90
        # min_size_score = 90
        # max_size_score = 100 - 110 = 0
        # size_score = (0 + 90 + 90 + 0) / 4 = 45
        # Retrieval Score:
        # (0.8 * 100 * 0.3333) + (0.75 * 100 * 0.3333) + (0.77 * 100 * 0.3333) ≈ 26.66 + 25 + 25.67 ≈ 77.33
        # Final Score = (45 * 0.4) + (77.33 * 0.6) = 18 + 46.4 = 64.4
        self.assertAlmostEqual(scores['Percentile']['articles'], 64.4, places=1)
        
        # Example for StdDeviation -> news
        # Size Score:
        # mean_score = 100 - 115 = 0
        # std_dev_score = 100 - 12 = 88
        # min_size_score = 95
        # max_size_score = 100 - 135 = 0
        # size_score = (0 + 88 + 95 + 0) / 4 = 45.75
        # Retrieval Score:
        # (0.88 * 100 * 0.3333) + (0.83 * 100 * 0.3333) + (0.85 * 100 * 0.3333) ≈ 29.33 + 27.66 + 28.33 ≈ 85.32
        # Final Score = (45.75 * 0.4) + (85.32 * 0.6) ≈ 18.3 + 51.19 ≈ 69.49
        self.assertAlmostEqual(scores['StdDeviation']['news'], 69.49, places=1)

if __name__ == '__main__':
    unittest.main()
