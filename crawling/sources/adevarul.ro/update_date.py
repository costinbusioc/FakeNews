import pandas as pd

data = pd.read_csv("adevarul.csv")

dates = data['date_published']
nr_rows = data.shape[0]

for i in range(nr_rows):

    year, month, day = data.loc[i, 'date_published'].split('-')
    if int(month) < 10:
        month = '0' + month
    if int(day) < 10:
        day = '0' + day

    data.at[i, 'date_published'] = f'{year}-{month}-{day}'

data.to_csv('adevarul.csv')
