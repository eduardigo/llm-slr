import os, pandas as pd

filename = './results/report.xlsx'
result_filename = './results/analysis.xlsx'
metrics = ['train_f1', 'train_roc_auc', 'f1', 'precision', 'recall', 'roc_auc', 'excluded', 'missed']

with pd.ExcelWriter(filename) as writer:
  df = pd.DataFrame()

  for metric in metrics:
    sheet = pd.read_excel(result_filename, sheet_name=metric, index_col=0)
    df['%s_mean' % (metric)] = sheet.mean(axis=0)

  df.to_excel(writer, sheet_name='summary')


  df_themes = pd.DataFrame()
  for theme in os.listdir('./bibs'):
    indexes = [ '%s-%d' % (theme, i) for i in range(3) ]

    for metric in metrics:
      sheet = pd.read_excel(result_filename, sheet_name=metric, index_col=0)
      themes_data = sheet.loc[indexes, :]
      df_themes['%s-%s-mean' % (theme, metric)] = themes_data.mean(axis=0)

  df_themes.T.to_excel(writer, sheet_name='by themes')
