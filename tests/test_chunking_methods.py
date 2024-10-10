import unittest
from src.chunking_methods.percentile import percentile_chunk
from src.chunking_methods.std_deviation import std_deviation_chunk
from chunking_methods.interquartile import interquantile_chunk
from src.chunking_methods.gradient import gradient_chunk

class TestChunkingMethods(unittest.TestCase):
    def setUp(self):
        self.text = "This is a sample text for testing chunking methods. " * 50  # >500 tokens

    def test_percentile_chunk(self):
        chunks = percentile_chunk(self.text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 1)

    def test_std_deviation_chunk(self):
        chunks = std_deviation_chunk(self.text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 1)

    def test_interquantile_chunk(self):
        chunks = interquantile_chunk(self.text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 1)

    def test_gradient_chunk(self):
        chunks = gradient_chunk(self.text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 1)

if __name__ == '__main__':
    unittest.main()
