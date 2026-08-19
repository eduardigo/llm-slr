import numpy as np, sys, os, pandas as pd

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from config import get_slr_files, get_classifier_pipeline, get_extractor, get_filters
from util.bib_loader import load
from util.years_split import YearsSplit
from util.text_filter import FilterComposite


if (len(sys.argv) < 2):
    print('second argument missing: SLR theme (games|slr|illiterate|pair|mdwe|testing|ontologies|xbi)')
    sys.exit(1)

if (len(sys.argv) < 3):
    print('third argument missing: classifier (svm|dt|rf|lsvm) or some transformers card-model (hugging-face)')
    sys.exit(1)

if (len(sys.argv) < 4):
    print('forth argument missing: titles only?')
    sys.exit(1)

if (len(sys.argv) < 5):
    print('fifth argument missing: extrator (tfidf,embeddings_glove,embeddings_se) or some transformers tokenizer (hugging-face)')
    sys.exit(1)

if (len(sys.argv) < 6):
    print('sixth argument missing: selector (selectkbest, truncatedsvd)')
    sys.exit(1)

_, theme, classifier_name, titles, extractor_name, selector_name = sys.argv
titles = True if titles == 'true' else False
embedding_filename = './embeddings/glove.6B.200d.txt' if extractor_name == 'embeddings_glove' else './embeddings/SO_vectors_200.bin'

met = ['train_f1', 'train_roc_auc', 'f1', 'precision', 'recall', 'roc_auc', 'excluded', 'missed']
scores = {}
for m in met: scores[m] = []

print('- Loading SLR data: %s' % (theme))
slr_files = get_slr_files(theme)
X, y, years = load(slr_files, titles_only=titles)

y_pred = []
y_true = []
y_pred_prob = []

X = np.array(X)
y = np.array(y)


print('- Preprocessing data (removing stopwords and lemmatizer)')
filters = get_filters(extractor_name)
preprocessor = FilterComposite(filters=filters)
X = np.array(preprocessor.fit_transform(X))

print('- Running Cross-Validation with %s' % (classifier_name))
kfold = YearsSplit(n_split=3, years=years)
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('   - Extracting features run...')
    extractor_pipeline = get_extractor(extractor_name, embeddings_filename=embedding_filename)
    X_train_ext = extractor_pipeline.fit_transform(X_train)
    X_test_ext = extractor_pipeline.transform(X_test)

    print('   - Grid Search run...')
    pipeline = get_classifier_pipeline(classifier_name, selector_name)
    pipeline.fit(X_train_ext, y_train)
    y_train_pred = (pipeline.predict(X_train_ext)).tolist()

    pred = (pipeline.predict(X_test_ext)).tolist()
    y_pred = y_pred + pred
    y_true = y_true + y_test.tolist()

    true_class_index = list(pipeline.classes_).index(1)
    y_prob = pipeline.predict_proba(X_train_ext)[:, true_class_index]
    precision, recall, thresholds = metrics.precision_recall_curve(y_train, y_prob)

    ind = -1
    for i in recall.tolist():
      if i == 1: ind += 1
      else: break

    threshold = min(thresholds[ind], 0.5)

    y_test_prob = (pipeline.predict_proba(X_test_ext)[:, true_class_index])
    y_test_prob = [ 0 if prob < threshold else 1 for prob in y_test_prob.tolist() ]
    y_pred_prob = y_pred_prob + y_test_prob

    matrix = metrics.confusion_matrix(y_test, y_test_prob)
    excluded = matrix[0, 0] / (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0])
    missed = matrix[1, 0] / (matrix[1, 1] + matrix[1, 0])

    scores['train_f1'].append(metrics.f1_score(y_train, y_train_pred))
    scores['train_roc_auc'].append(metrics.roc_auc_score(y_train, y_train_pred))
    scores['f1'].append(metrics.f1_score(y_test.tolist(), pred))
    scores['recall'].append(metrics.recall_score(y_test.tolist(), pred))
    scores['precision'].append(metrics.precision_score(y_test.tolist(), pred))
    scores['roc_auc'].append(metrics.roc_auc_score(y_test.tolist(), pred))
    scores['excluded'].append(excluded)
    scores['missed'].append(missed)


print('')
print('')
print('====================================')
print('=        General accuracy          =')
print('====================================')
print(metrics.classification_report(y_true, y_pred))
print('            ROC-AUC: %f' % (metrics.roc_auc_score(y_true, y_pred)))
print('====================================')
print('Configuring the activation threshold')
print('====================================')
print(metrics.classification_report(y_true, y_pred_prob))
print('            ROC-AUC: %f' % (metrics.roc_auc_score(y_true, y_pred_prob)))
print('====================================')
print('         SLR Updates metrics')
print('====================================')
print('Excluded percentage: %f' % (sum(scores['excluded']) / len(scores['excluded'])))
print('  Missed percentage: %f' % (sum(scores['missed']) / len(scores['missed'])))

results_filename = './results/analysis.xlsx'
model_name = '%s-%s-%s-%s' % (extractor_name, classifier_name, selector_name, titles)

if os.path.exists(results_filename):
  prev = {}
  for m in met:
    prev[m] = pd.read_excel(results_filename, sheet_name=m, index_col=0)

  with pd.ExcelWriter(results_filename) as writer:
    for m in met:
      series = pd.Series(scores[m])
      series.index = [ '%s-%d' % (theme, i) for i in range(len(scores[m]))]

      if series.index[0] not in prev[m].index:
        empty_df = pd.DataFrame(index=series.index, columns=prev[m].columns)
        prev[m] = pd.concat([prev[m], empty_df])

      prev[m].loc[series.index, model_name] = series
      prev[m].to_excel(writer, sheet_name=m)

else:
  with pd.ExcelWriter(results_filename) as writer:
    for m in met:
      df = pd.DataFrame()
      series = pd.Series(scores[m])
      series.index = [ '%s-%d' % (theme, i) for i in range(len(scores[m]))]

      df[model_name] = series
      df.to_excel(writer, sheet_name=m)


sys.exit(0)
