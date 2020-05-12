import numpy as np

import io
import csv

import argparse
from collections import Counter

from gensim.models.keyedvectors import KeyedVectors

from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser
from rb.processings.encoders.bert import BertWrapper
from tensorflow import keras

from helpers import write_csv

args = None


class DocVectorGeneration:
    MAX_NB_WORDS = 1000000
    ENGLISH = "en"

    nlp_ro = SpacyParser.get_instance().get_model(Lang.RO)

    word_embeddings_model = "word2vec.model"
    embeddings_dim = 300

    def __init__(
        self,
        input_file,
        vector_repr,
        small_run=True,
        non_freq_nouns=False,
    ):
        self.input_file = input_file
        self.output_file = input_file.split('.')[0] + '_' + vector_repr + '_vectors.csv'

        w2v_needed = vector_repr != 'bert'
        if w2v_needed:
            self.load_w2v_model()

        if non_freq_nouns:
            self.non_freq_nouns = []
            with open(self.nouns_file, "r") as f:
                for line in f.readlines():
                    self.non_freq_nouns.append(line.strip())

        self.load_data(small_run)

    def load_data(self, small_run):
        global sources_dict

        csv.field_size_limit(10000000)

        texts, titles, categories = [], [], []
        cnt_skipped_examples = 0

        with io.open(
            self.input_file, "r", encoding="utf-8", errors="replace"
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

                    except:
                        cnt_skipped_examples += 1
                        continue

                    category = line[3]

                    titles.append(title)
                    texts.append(text)
                    categories.append(category)

            print(
                "dataset loaded, skipped examples {} from total of {}, remaining {}".format(
                    cnt_skipped_examples, index, len(categories)
                )
            )

        if args.small_run == True:
            titles = titles[:100]
            texts = texts[:100]
            categories = categories[:100]

        print("labels distribution, all: ")
        counters_categories = Counter(categories)
        print(counters_categories)

        self.categories = categories
        self.titles = titles
        self.texts = texts

        return titles, texts

    def load_w2v_model(self):
        print("Loading model")
        self.word_emb = KeyedVectors.load_word2vec_format(
            self.word_embeddings_model, binary=False
        )
        print("Model loaded")

    def preprocess_text(self, texts, nouns_only=False, non_freq_nouns=False):
        docs = [self.nlp_ro(text) for text in texts]
        docs = [[token for token in doc if not token.is_stop] for doc in docs]

        if nouns_only:
            docs = list(
                map(
                    lambda doc: list(
                        filter(lambda token: token.tag_.startswith("N"), doc)
                    ),
                    docs,
                )
            )
        elif non_freq_nouns:
            docs = list(
                map(
                    lambda doc: list(
                        filter(
                            lambda token: token.tag_.startswith("Np")
                            or str(token.lemma_).lower().strip() in self.non_freq_nouns,
                            doc,
                        )
                    ),
                    docs,
                )
            )
            # docs = list(map(lambda doc: list(filter(lambda token: str(token.lemma_).lower().strip() in self.non_freq_nouns, doc)), docs))
        else:
            docs = list(
                map(
                    lambda doc: list(
                        filter(lambda token: not token.tag_.startswith("P"), doc)
                    ),
                    docs,
                )
            )

        # Lower each word and lemmatize
        for i, doc in enumerate(docs):
            docs[i] = [token.lemma_.lower() for token in doc]

        # docs = list(map(lambda doc: " ".join([token.lemma_.lower().strip() for token in doc]), docs))

        return docs

    def process_title_text(self, titles, texts):
        processed_titles = self.preprocess_text(titles)
        print("Titles processed")
        processed_texts = self.preprocess_text(texts)
        print("Texts processed")

        return processed_titles, processed_texts

    def compute_w2v_avg(self, texts):
        avg_w2v = []

        for doc in texts:
            sum_doc = np.zeros(self.embeddings_dim)
            for token in doc:
                word_v = np.zeros(self.embeddings_dim)
                if token in self.word_emb:
                    word_v = self.word_emb[token]
                sum_doc = np.add(sum_doc, word_v)

            if len(doc):
                avg_w2v.append(sum_doc / len(doc))
            else:
                avg_w2v.append(sum_doc)

        return avg_w2v

    def text_to_vector(self, titles, texts, vector_repr):
        if vector_repr == "bert":
            bert_wrapper = BertWrapper(Lang.RO, max_seq_len=128)
            inputs, bert_output = bert_wrapper.create_inputs_and_model()
            cls_output = bert_wrapper.get_output(bert_output, "cls")
            print('dupa class output')

            model = keras.Model(inputs=inputs, outputs=[cls_output])
            bert_wrapper.load_weights()
            print('weights loaded')
            feed_inputs = bert_wrapper.process_input(texts)
            print('inputs processed')
            predictions = model.predict(feed_inputs, batch_size=32)
            print('predictionssssss')
            return predictions
        else:
            if vector_repr == "w2v_titles":
                titles = self.preprocess_text(titles)
                return self.compute_w2v_avg(titles)
            elif vector_repr == "w2v_texts":
                texts = self.preprocess_text(texts)
                return self.compute_w2v_avg(texts)
            elif vector_repr == "w2v_titles_texts":
                titles, texts = self.process_title_text(titles, texts)
                vector_titles = self.compute_w2v_avg(titles)
                vector_texts = self.compute_w2v_avg(texts)
                return [sum(e) / len(e) for e in zip([vector_titles, vector_texts])]

    def vectorize_text(self, vector_repr):
        print("Inainte de vectori")
        vectors = self.text_to_vector(self.titles, self.texts, vector_repr)

        cols_name = ["Title", "Text", "Category", "Vector"]
        cols = [self.titles, self.texts, self.categories, vectors]
        write_csv(self.output_file, cols_name, cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        "--small_run", dest="small_run", action="store_true", default=False
    )
    parser.add_argument(
        "--vector_repr",
        dest="vector_repr",
        action="store",
        type=str,
        default="bert",
        choices=["w2v_titles", "w2v_texts", "w2v_titles_texts", "bert"],
        help="Vector text represenation",
    )
    parser.add_argument(
        "--nouns_only", dest="nouns_only", action="store_true", default=False
    )
    parser.add_argument(
        "--non_freq_nouns", dest="non_freq_nouns", action="store_true", default=False
    )

    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, "->", args.__dict__[k])

    in_file = "category_news.csv"
    doc_vector_generation = DocVectorGeneration(in_file, args.vector_repr, small_run=args.small_run)
    doc_vector_generation.vectorize_text(args.vector_repr)
