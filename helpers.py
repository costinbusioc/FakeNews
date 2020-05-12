import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

def write_csv(filename, col_names, cols):
    df = pd.DataFrame(cols)
    df = df.transpose()

    with open(filename, 'w', encoding='utf-8') as f:
            df.to_csv(f, header=col_names)


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

