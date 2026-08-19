from unittest import TestCase
from unittest.mock import MagicMock, Mock, call

from util.embedding_vectorizer import AverageEmbeddingVectorizer

class AverageEmbeddingVectorizerTest (TestCase):


  def setUp(self):
    self.kv_mock = MagicMock()
    self.vectorizer = AverageEmbeddingVectorizer(self.kv_mock)


  def test_transform_calls_get_mean_vector_once (self):
    self.kv_mock.get_mean_vector.return_value = [1, 2, 3]

    result = self.vectorizer.transform(['something is not right'])

    self.kv_mock.get_mean_vector.assert_called_with(['something', 'is', 'not', 'right'])
    self.assertEqual([[1, 2, 3]], result)


  def test_transform_calls_get_mean_vector_for_each_row (self):
    def side_effect (arg):
      mapping = {
        'one-more-thing': [4, 5, 6, 7, 8],
        'here-is-an-apple': [9, 10, 11, 12, 13]
      }
      return mapping['-'.join(arg)]

    self.kv_mock.get_mean_vector.side_effect = side_effect

    result = self.vectorizer.transform(['one more thing', 'here is an apple'])

    self.kv_mock.get_mean_vector.assert_has_calls([
      call(['one', 'more', 'thing']), call(['here', 'is', 'an', 'apple'])])
    self.assertEqual([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]], result)


  def test_transform_does_not_call_get_mean_vector (self):
    result = self.vectorizer.transform([])

    self.kv_mock.get_mean_vector.assert_not_called()
    self.assertEqual([], result)


  def test_vectorize_implements_fit_interface (self):
    X_stub, y_stub = [], []
    result = self.vectorizer.fit(X_stub, y_stub)
    self.assertEqual(self.vectorizer, result)


  def test_vectorize_implements_fit_transform_interface (self):
    X_stub, y_stub = [], []
    self.vectorizer.transform = Mock(return_value=[[1, 2], [3, 4]])
    self.vectorizer.fit = Mock(return_value=self.vectorizer)
    result = self.vectorizer.fit_transform(X_stub, y_stub)

    self.vectorizer.transform.assert_called_with(X_stub)
    self.vectorizer.fit.assert_called_with(X_stub, y_stub)

    self.assertEqual([[1, 2], [3, 4]], result)


