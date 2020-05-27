import json
import operator
from collections import defaultdict

texts = defaultdict(int)
titles = defaultdict(int)

with open('../dumped_news.json') as json_data:
    d = json.load(json_data)
    for obj in d['articles_list']:
        texts[(obj['title'], obj['text'])] = texts[(obj['title'], obj['text'])] + 1
        titles[obj['title']] = titles[obj['title']] + 1

texts = {k: v for k, v in sorted(texts.items(), key=lambda item: item[1], reverse=True)}
for key, value in texts.items():
    if value > 1:
        print(f"{value} - {titles[key[0]]}: {key[0]}")
