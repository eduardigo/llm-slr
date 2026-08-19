from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class FilterComposite ():

    def __init__ (self, filters=[]):
        self._filters = filters

    def fit (self, X, y=None):
        return self

    def transform (self, X, y=None):
        X_token = [ word_tokenize(row.lower()) for row in X ]
        X_filtered = []
        for tokens in X_token:
            filtered_tokens = tokens
            for f in self._filters:
                filtered_tokens = f.filter(filtered_tokens)
            X_filtered.append(filtered_tokens)
        return [ ' '.join(row) for row in  X_filtered]

    def fit_transform (self, X, y=None):
        return self.fit(X).transform(X)

    def get_params (self, deep=False):
        return { 'filters': self._filters }

    def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self


class StopwordsFilter ():

    def filter (self, tokens):
        return [
            token for token in tokens
                  if token not in stopwords.words('english')]


class LemmatizerFilter ():

    def __init__ (self):
        self._lemmatizer = WordNetLemmatizer()

    def filter (self, tokens):
        tags = pos_tag(tokens)
        return [ self._lemmatizer.lemmatize(tag[0], pos=tag[1][0].lower())
                    if tag[1][0].lower() in ('a', 'n', 'v', 'r')
                    else self._lemmatizer.lemmatize(tag[0])
                    for tag in tags ]
