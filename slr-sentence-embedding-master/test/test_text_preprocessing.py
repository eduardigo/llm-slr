from unittest import TestCase
from unittest.mock import Mock

from util.text_filter import FilterComposite, StopwordsFilter, LemmatizerFilter

class FilterCompositeTest (TestCase):

    def test_filter_implements_pipeline (self):
        X = []
        f = StopwordsFilter()
        fil = FilterComposite([f])
        s = fil.fit(X)
        res = fil.transform(X)

        self.assertEqual(fil, s)
        self.assertEqual([], res)

        s = fil.fit(X, [])
        res = fil.transform(X, [])

        self.assertEqual(fil, s)
        self.assertEqual([], res)

        self.assertEqual({ 'filters': [f] }, fil.get_params())
        self.assertEqual({ 'filters': [f] }, fil.get_params(False))


    def test_fit_transform(self):
        X = []
        fil = FilterComposite([StopwordsFilter()])
        fil.fit = Mock(return_value=fil)
        fil.transform = Mock(return_value=[1, 2, 3])

        res = fil.fit_transform(X)

        fil.fit.assert_called_once_with(X)
        fil.transform.assert_called_once_with(X)

        self.assertEqual([1, 2, 3], res)

        fil.fit = Mock(return_value=fil)
        fil.transform = Mock(return_value=[1, 2, 3])

        res = fil.fit_transform(X, [])

        fil.fit.assert_called_once_with(X)
        fil.transform.assert_called_once_with(X)

        self.assertEqual([1, 2, 3], res)


    def test_transform_removes_stopwords (self):
        X = ['Some other area of testing a single approach with twinkles',
             'Another Super Example of Fish under the Table with ketchup']
        fil = FilterComposite([StopwordsFilter()])

        res = fil.transform(X)

        self.assertEqual([
            'area testing single approach twinkles',
            'another super example fish table ketchup'], res)


    def test_transform_reduces_words_to_lemmas (self):
        X = ['Testing an approach, based on typing on the keyboard',
             'Evaluating multiple strategies for swimming in the pool']
        fil = FilterComposite([LemmatizerFilter()])

        res = fil.transform(X)

        self.assertEqual([
            'test an approach , base on type on the keyboard',
            'evaluate multiple strategy for swim in the pool'], res)


    def test_transform_reduces_words_to_lemmas (self):
        X = ['Some other area of testing a single approach with twinkles',
             'Another Super Example of Fish under the Table with ketchup',
             'Testing an approach, based on typing on the keyboard',
             'Evaluating multiple strategies for swimming in the pool']
        fil = FilterComposite([StopwordsFilter(), LemmatizerFilter()])

        res = fil.transform(X)

        self.assertEqual([
            'area test single approach twinkle',
            'another super example fish table ketchup',
            'test approach , base type keyboard',
            'evaluate multiple strategy swim pool'], res)
