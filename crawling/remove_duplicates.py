from sklearn.utils import shuffle
import os
import pandas as pd
import glob

file_name = 'romanian_news_2020.csv'

df = pd.read_csv(file_name)
df = df.drop_duplicates(subset=['title', 'source', 'date_published'], keep='first')
df = shuffle(df)
df.to_csv(file_name)

