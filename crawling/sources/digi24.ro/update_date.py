import pandas as pd

data = pd.read_csv("digi24.csv")

dates = data['date_published']
nr_rows = data.shape[0]
print(nr_rows)

for i in range(nr_rows):
    real_date = data.loc[i, 'date_published'].split('T')[0]
    data.loc[i, 'date_published'] = real_date

data.to_csv('digi24.csv')
