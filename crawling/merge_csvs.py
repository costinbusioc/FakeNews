from sklearn.utils import shuffle
import os
import pandas as pd
import glob

root_dir = './sources'
all_csvs = []

for subdir, dirs, files in os.walk(root_dir):
    all_csvs += glob.glob(f'{subdir}/*.csv')

df = pd.concat(pd.read_csv(f, names = ['source', 'title', 'text', 'url',
'date_published', 'authors'], header=0) for f in all_csvs)
df_deduplicated = df.drop_duplicates()
df_deduplicated = shuffle(df_deduplicated)
df_deduplicated.to_csv("romanian_news.csv")

