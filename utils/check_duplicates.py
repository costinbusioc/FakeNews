import json
import operator
from collections import defaultdict

titles = defaultdict(int)

with open('../dumped_news.json') as json_data:
    d = json.load(json_data)
    for obj in d['articles_list']:
        titles[obj['title']] = titles[obj['title']] + 1

titles = {k: v for k, v in sorted(titles.items(), key=lambda item: item[1], reverse=True)}
for key, value in titles.items():
    if value > 1:
        print(f"{value}: {key}")
        print('\n\n')
