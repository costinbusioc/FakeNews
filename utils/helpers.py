import io
import csv
import numpy as np
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors

def write_csv(filename, col_names, cols):
    df = pd.DataFrame(cols)
    df = df.transpose()

    with open(filename, 'w', encoding='utf-8') as f:
        df.to_csv(f, header=col_names)


def remove_duplicates(input_file, output_file):
    df = pd.read_csv(input_file)
    print(df.shape)

    df = df.drop_duplicates(subset=['Titles', 'Source'], keep='first')
    print(df.shape)

    df = shuffle(df)
    df.to_csv(output_file)


def load_w2v_model(word_embeddings_model):
    print("Loading model")
    word_emb = KeyedVectors.load_word2vec_format(
        word_embeddings_model, binary=False
    )
    print("Model loaded")
    return word_emb

def compute_w2v_avg(texts, embeddings_dim, word_emb):
    avg_w2v = []

    for doc in texts:
        sum_doc = np.zeros(embeddings_dim)
        for token in doc:
            word_v = np.zeros(embeddings_dim)
            if token in word_emb:
                word_v = word_emb[token]
            sum_doc = np.add(sum_doc, word_v)

        if len(doc):
            avg_w2v.append(sum_doc / len(doc))
        else:
            avg_w2v.append(sum_doc)

    return avg_w2v


def load_data(input_file, with_vec=False, small_run=True):
    csv.field_size_limit(10000000)

    texts, titles, categories = [], [], []
    urls, sources, dates = [], [], []
    vectors = []
    cnt_skipped_examples = 0

    with io.open(
            input_file, "r", encoding="utf-8", errors="replace"
    ) as csv_file:
        index = 0
        csv_reader = csv.reader(csv_file, delimiter=",")

        for line in csv_reader:
            index += 1
            # skip first x examples
            if index > 1:
                try:
                    title = line[1].strip()
                    text = line[2].strip()
                    category = line[3].strip()

                except:
                    cnt_skipped_examples += 1
                    continue

                url = line[6].strip()
                source = line[4].strip()
                date = line[5].strip()

                titles.append(title)
                texts.append(text)
                categories.append(category)
                urls.append(url)
                sources.append(source)
                dates.append(date)

                if with_vec:
                    vector = line[7].strip()
                    vector = vector.split()
                    if vector[0] == '[':
                        vector = vector[1:]
                    if vector[0][0] == "[":
                        vector[0] = vector[0][1:]
                    if vector[-1][-1] == "]":
                        vector[-1] = vector[-1][:-1]
                    vector = list(map(float, vector))

                    if len(vector) != 768:
                        print(len(vector))
                    vector = np.array(vector, dtype=np.float32)
                    vectors.append(vector)

        print(
            "dataset loaded, skipped examples {} from total of {}, remaining {}".format(
                cnt_skipped_examples, index, len(categories)
            )
        )

    if small_run == True:
        titles = titles[:100]
        texts = texts[:100]
        categories = categories[:100]
        urls = urls[:100]
        sources = sources[:100]
        dates = dates[:100]
        vectors = vectors[:100]

    print("labels distribution, all: ")
    counters_categories = Counter(categories)
    print(counters_categories)

    result = (titles, texts, categories, urls, sources, dates)
    if with_vec:
        result = (titles, texts, categories, urls, sources, dates, vectors)

    return result


def main():
    pass


if __name__ == "__main__":
    main()
