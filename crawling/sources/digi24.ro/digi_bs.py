from datetime import date
import requests
from bs4 import BeautifulSoup

import csv
import json
import os
import glob
import pandas as pd

base_url = "https://www.digi24.ro/stiri/actualitate/politica"

root_url = "https://digi24.ro"

source = "digi24.ro"

number_pages = 120

authors_list = []
date_list = []
title_list = []
text_list = []
url_list = []
source_list = []

empty_articles = 0

nr = 0

for i in range(number_pages):
    print('----------------------------')
    print(f"Page={i}\n")

    curr_url = f"{base_url}?p={i}.html"

    page = requests.get(curr_url)
    soup = BeautifulSoup(page.content, "html.parser")

    articles = soup.find_all('article')

    for article in articles:

        article_url = article.find('h4', class_='article-title')
        if not article_url:
            article_url = article.find('h2', class_='article-title')

        article_url = article_url.find('a')['href']

        article_url = f"{root_url}{article_url}"

        article_page = requests.get(article_url)
        article_soup = BeautifulSoup(article_page.content, "html.parser")

        title = article_soup.find('div', class_='col-8 col-md-9 col-sm-12').find('h1').text.strip()
        author = ""

        date_publish = article_soup.find('div', class_='author').find('time')['datetime']
        #date_publish = date.fromtimestamp(int(date_publish))
        #date_publish = f"{date_publish.year}-{date_publish.month}-{date_publish.day}"

        article_text = ""

        text = article_soup.find('div', class_='col-10 col-sm-12')
        text = text.find_all('p')
        for t in text:
            article_text += " " + t.text.strip()

        print(title)

        '''
        print(author)
        print(date_publish)
        print(article_url)
        print(article_text)
        '''

        nr += 1

        if not title or not article_text or not date_publish:
            empty_articles += 1
            continue

        authors_list.append(author.replace(';', '---'))
        date_list.append(date_publish)
        title_list.append(title.replace(';', ' '))
        text_list.append(article_text.replace(';', ' '))
        url_list.append(article_url)
        source_list.append(source)

    print(nr, i)


print(f"Empty articles in {source_list[0]}: {empty_articles}")

cols = [source_list, title_list, text_list, url_list, date_list, authors_list]
col_names = ['source', 'title', 'text', 'url', 'date_published', 'authors']

df = pd.DataFrame(cols)
df = df.transpose()

with open('digi24.csv', 'w', encoding='utf-8') as f:
    df.to_csv(f, header=col_names)

