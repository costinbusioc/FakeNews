from datetime import date
import requests
from bs4 import BeautifulSoup

import csv
import json
import os
import glob
import pandas as pd

path = '.'

authors_list = []
date_list = []
title_list = []
text_list = []
url_list = []
source_list = []

empty_articles = 0

for filename in glob.glob(os.path.join(path, '*.json')):
    with open(filename, encoding='utf-8', mode='r') as curr_json:
        data = curr_json.read()
        data = json.loads(data)

        authors = data.get('authors', None)
        date_publish = data.get('date_publish', None)
        date_modify = data.get('date_modify', None)
    
        date = date_modify if date_modify else date_publish

        url = data.get('url')

        #source = data.get('source_domain')
        source = "stiripesurse.ro"
        text = data.get('text')
        title = data.get('title').replace(';', '.')

        if not text:
            if not url:
                empty_articles += 1
                continue

            article_page = requests.get(url)
            article_soup = BeautifulSoup(article_page.content, "html.parser")

            content = article_soup.find('section', class_='article-content')
            content = content.find_all('p')
            text = ""

            for t in content:
                text += " " + t.text.strip()      

        if not title or not text or not date:
            empty_articles += 1
            continue

        authors = None if not authors else "---".join(authors)

        authors_list.append(authors)
        date_list.append(date)
        title_list.append(title)
        text_list.append(text)
        url_list.append(url)
        source_list.append(source)


print(f"Empty articles in {source_list[0]}: {empty_articles}")


cols = [source_list, title_list, text_list, url_list, date_list, authors_list]
col_names = ['source', 'title', 'text', 'url', 'date_published', 'authors']

df = pd.DataFrame(cols)
df = df.transpose()

with open('stiripesurse.csv', 'w', encoding='utf-8') as f:
    df.to_csv(f, header=col_names)

