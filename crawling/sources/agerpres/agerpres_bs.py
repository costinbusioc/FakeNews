from datetime import date
import requests
from bs4 import BeautifulSoup

import csv
import json
import os
import glob
import pandas as pd

base_url = "https://www.agerpres.ro/politica/"

root_url = "https://www.agerpres.ro"

source = "agerpres.ro"

number_pages = 200

authors_list = []
date_list = []
title_list = []
text_list = []
url_list = []
source_list = []

empty_articles = 0

nr = 0

for i in range(1, number_pages):
    print('-------------------')
    print(f"Page={i}")

    curr_url = f"{base_url}page/{i}"

    page = requests.get(curr_url)
    soup = BeautifulSoup(page.content, "html.parser")

    articles = soup.find_all('article', class_='unit_news shadow p_r')
    for article in articles:
        article_url = article.find('div', class_='title_news').find('h2').find('a')['href']
        article_url = f"{root_url}{article_url}"
        
        title = article.find('div', class_='title_news').find('h2').text.strip()
        date_publish = article.find('div', class_='publish_time').find('time')['title']

        article_page = requests.get(article_url)
        article_soup = BeautifulSoup(article_page.content, "html.parser")

        author = ''


        article_text = ''
        text = article_soup.find('div', class_='description_articol')
        text = text.find_all('p')
        for t in text:
            article_text += "\n" + t.text.strip()

        nr += 1

        if not title or not article_text or not date_publish:
            empty_articles += 1
            continue

        print(title)

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

with open('agerpres.csv', 'w', encoding='utf-8') as f:
    df.to_csv(f, header=col_names)

