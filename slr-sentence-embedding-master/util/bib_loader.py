import codecs, bibtexparser

def load(file_list, titles_only=False):
    dataset = []
    for f in file_list:
        with codecs.open(f, 'r', encoding='utf-8') as bib_f:
            db = bibtexparser.load(bib_f)
            for bib_index, entry in enumerate(db.entries, start=0):
                label = 1 if entry['inserir'] == 'true' else 0
                title = entry['title']
                abstract = entry['abstract']
                if (titles_only):
                    content = u'%s' % (title)
                else:
                    content = u'%s\n%s' % (title, abstract)
                year = int(entry['year'])
                dataset.append({
                    'content': content,
                    'label': label,
                    'year': year
                })

    def sort_criteria(row):
        return row['year']
    dataset.sort(key=sort_criteria)

    X = [row['content'] for row in dataset]
    y = [row['label'] for row in dataset]
    years = [row['year'] for row in dataset]
    return X, y, years
