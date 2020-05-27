import json
from collections import defaultdict

from helpers import write_csv

texts = defaultdict(str)
titles = defaultdict(str)
categories = defaultdict(str)
url = defaultdict(str)
source = defaultdict(str)
date = defaultdict(str)

with open("../predictions.json") as json_data:
    d = json.load(json_data)
    for obj in d:
        categories[obj["id"]] = obj["category"]
print('Done predictions')

count = 0
with open("../dumped_news.json") as json_data:
    d = json.load(json_data)
    for obj in d["articles_list"]:
        titles[obj["id"]] = obj["title"]
        texts[obj["id"]] = obj["text"]
        url[obj["id"]] = obj["url"]
        source[obj["id"]] = obj["source_domain"]

        try:
            date[obj["id"]] = obj["date"]
        except:
            date[obj["id"]] = None
            print('Lipsa')
            count += 1

print(count)

print('Done dump')


texts_list = []
titles_list = []
categories_list = []
url_list = []
source_list = []
date_list = []

for key in categories.keys():
    if not texts[key] or not titles[key]:
        continue

    texts_list.append(texts[key])
    titles_list.append(titles[key])
    categories_list.append(categories[key])
    url_list.append(url[key])
    source_list.append(source[key])
    date_list.append(date[key])

print('Writing csv')
write_csv(
    "category_news.csv",
    ["Titles", "Texts", "Categories", "Url", "Source", "Date"],
    [titles_list, texts_list, categories_list, url_list, source_list, date_list],
)
