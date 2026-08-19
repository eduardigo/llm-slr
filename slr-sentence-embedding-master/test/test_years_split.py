from unittest import TestCase

from util.years_split import YearsSplit

class YearsSplitTest (TestCase):

    def test_years_split_separate_years (self):
        split = YearsSplit(n_split=3, years=[
            0,0,1,1,1,2,2,2,2,2,3,3,3,3,3])
        X = []
        y = []

        generator = split.split(X, y)
        train, test = next(generator)
        self.assertEqual(train, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(test, [10, 11, 12, 13, 14])

        train, test = next(generator)
        self.assertEqual(train, [0, 1, 2, 3, 4])
        self.assertEqual(test, [5, 6, 7, 8, 9])

        train, test = next(generator)
        self.assertEqual(train, [0, 1])
        self.assertEqual(test, [2, 3, 4])

        with self.assertRaises(StopIteration):
            next(generator)

    def test_years_split_should_group_years_with_less_than_5 (self):
        split = YearsSplit(n_split=3, years=[
            0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3])
        X = []
        y = []

        generator = split.split(X, y)
        train, test = next(generator)
        self.assertEqual(train, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(test, [10, 11, 12, 13, 14, 15, 16, 17])
