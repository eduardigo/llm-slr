import codecs, io

from unittest import TestCase
from unittest.mock import call, patch, Mock

from util.bib_loader import load


class BibLoaderTest(TestCase):

    def test_load_bibitem (self):
        bibtex_content = '''
@ARTICLE{I[1],
  inserir = {true},
  title = {abobrinha1},
  year = {2008},
  abstract = {abobrinha abstract},
}
        '''
        file_stub = Mock()
        file_stub.__enter__ = Mock(return_value=file_stub)
        file_stub.__exit__ = Mock()
        file_stub.read = Mock(return_value=bibtex_content)

        with patch.object(codecs, 'open', return_value=file_stub) as cod:
            X, y, years = load(['abobrinha.bib'])

        cod.assert_called_once_with('abobrinha.bib', 'r', encoding='utf-8')
        self.assertEqual(X, ['abobrinha1\nabobrinha abstract'])
        self.assertEqual(y, [1])
        self.assertEqual(years, [2008])

    def test_load_2_bibitems (self):
        bibtex_content = '''
@ARTICLE{I[2],
  inserir = {true},
  title = {abobrinha1},
  year = {2008},
  abstract = {abobrinha abstract},
}
@INPROCEEDINGS{E[2],
  inserir = {false},
  title = {umbrela},
  year = {2020},
  abstract = {another abstract},
}
        '''
        file_stub = Mock()
        file_stub.__enter__ = Mock(return_value=file_stub)
        file_stub.__exit__ = Mock()
        file_stub.read = Mock(return_value=bibtex_content)

        with patch.object(codecs, 'open', return_value=file_stub) as cod:
            X, y, years = load(['pepino.bib'])

        cod.assert_called_once_with('pepino.bib', 'r', encoding='utf-8')
        self.assertEqual(X, [
            'abobrinha1\nabobrinha abstract',
            'umbrela\nanother abstract'])
        self.assertEqual(y, [1, 0])
        self.assertEqual(years, [2008, 2020])

    def test_load_option_for_titles_only (self):
        bibtex_content = '''
@ARTICLE{I[2],
  inserir = {true},
  title = {abobrinha1},
  year = {2008},
  abstract = {abobrinha abstract},
}
@INPROCEEDINGS{E[2],
  inserir = {false},
  title = {umbrela},
  year = {2020},
  abstract = {another abstract},
}
        '''
        file_stub = Mock()
        file_stub.__enter__ = Mock(return_value=file_stub)
        file_stub.__exit__ = Mock()
        file_stub.read = Mock(return_value=bibtex_content)

        with patch.object(codecs, 'open', return_value=file_stub) as cod:
            X, y, years = load(['pepino.bib'], titles_only=True)

        cod.assert_called_once_with('pepino.bib', 'r', encoding='utf-8')
        self.assertEqual(X, [
            'abobrinha1',
            'umbrela'])
        self.assertEqual(y, [1, 0])
        self.assertEqual(years, [2008, 2020])

    def test_load_2_bibfiles (self):
        bibtex_content_1 = '''
@ARTICLE{I[2],
  inserir = {true},
  title = {abobrinha1},
  year = {2008},
  abstract = {abobrinha abstract},
}
@INPROCEEDINGS{E[2],
  inserir = {false},
  title = {umbrela},
  year = {2020},
  abstract = {another abstract},
}
        '''
        file_stub_1 = Mock()
        file_stub_1.__enter__ = Mock(return_value=file_stub_1)
        file_stub_1.__exit__ = Mock()
        file_stub_1.read = Mock(return_value=bibtex_content_1)
        bibtex_content_2 = '''
@ARTICLE{I[2],
  inserir = {false},
  title = {uva},
  year = {2020},
  abstract = {contos da uva},
}
@INPROCEEDINGS{E[2],
  inserir = {false},
  title = {note},
  year = {2021},
  abstract = {not cool},
}
        '''
        file_stub_2 = Mock()
        file_stub_2.__enter__ = Mock(return_value=file_stub_2)
        file_stub_2.__exit__ = Mock()
        file_stub_2.read = Mock(return_value=bibtex_content_2)
        files = [file_stub_1, file_stub_2]

        with patch.object(codecs, 'open', side_effect=files) as cod:
            X, y, years = load(['pepino.bib', 'abacaxi.bib'])

        cod.assert_has_calls([
            call('pepino.bib', 'r', encoding='utf-8'),
            call('abacaxi.bib', 'r', encoding='utf-8')])
        self.assertEqual(X, [
            'abobrinha1\nabobrinha abstract',
            'umbrela\nanother abstract',
            'uva\ncontos da uva',
            'note\nnot cool'])
        self.assertEqual(y, [1, 0, 0, 0])
        self.assertEqual(years, [2008, 2020, 2020, 2021])

    def test_load_2_bibfiles_and_sort_years (self):
        bibtex_content_1 = '''
@ARTICLE{I[2],
  inserir = {true},
  title = {abobrinha1},
  year = {2020},
  abstract = {abobrinha abstract},
}
@INPROCEEDINGS{E[2],
  inserir = {false},
  title = {umbrela},
  year = {2019},
  abstract = {another abstract},
}
        '''
        file_stub_1 = Mock()
        file_stub_1.__enter__ = Mock(return_value=file_stub_1)
        file_stub_1.__exit__ = Mock()
        file_stub_1.read = Mock(return_value=bibtex_content_1)
        bibtex_content_2 = '''
@ARTICLE{I[2],
  inserir = {false},
  title = {uva},
  year = {2008},
  abstract = {contos da uva},
}
@INPROCEEDINGS{E[2],
  inserir = {true},
  title = {note},
  year = {2001},
  abstract = {not cool},
}
        '''
        file_stub_2 = Mock()
        file_stub_2.__enter__ = Mock(return_value=file_stub_2)
        file_stub_2.__exit__ = Mock()
        file_stub_2.read = Mock(return_value=bibtex_content_2)
        files = [file_stub_1, file_stub_2]

        with patch.object(codecs, 'open', side_effect=files) as cod:
            X, y, years = load(['pepino.bib', 'abacaxi.bib'])

        cod.assert_has_calls([
            call('pepino.bib', 'r', encoding='utf-8'),
            call('abacaxi.bib', 'r', encoding='utf-8')])
        self.assertEqual(X, [
            'note\nnot cool',
            'uva\ncontos da uva',
            'umbrela\nanother abstract',
            'abobrinha1\nabobrinha abstract'])
        self.assertEqual(y, [1, 0, 0, 1])
        self.assertEqual(years, [2001, 2008, 2019, 2020])
