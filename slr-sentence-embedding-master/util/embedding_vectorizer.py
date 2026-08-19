import numpy as np

from nltk.tokenize import word_tokenize

class AverageEmbeddingVectorizer():

    def __init__(self, kv=None):
        self._kv = kv

    def fit (self, X, y):
        return self

    def fit_transform (self, X, y):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        sentence_tokens = [ word_tokenize(sentence) for sentence in X ]

        return [ self._kv.get_mean_vector(tokens) for tokens in sentence_tokens ]

    def get_params (self, deep=False):
        return { 'kv': self._kv }

    def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self


