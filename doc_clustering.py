import numpy as np
import sys

from sklearn.cluster import AffinityPropagation, Birch, DBSCAN

from scipy.spatial.distance import cdist

from gensim.models.keyedvectors import KeyedVectors

from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser
from rb.processings.encoders.bert import BertWrapper
from tensorflow import keras

class DocClustering:
    MAX_NB_WORDS = 1000000
    ENGLISH = "en"

    nlp_ro = SpacyParser.get_instance().get_model(Lang.RO)

    word_embeddings_model = "word2vec.model"
    embeddings_dim = 300

    def __init__(self, w2v_needed=False, non_freq_nouns=False):

        if w2v_needed:
            self.load_w2v_model()

        if non_freq_nouns:
            self.non_freq_nouns = []
            with open(self.nouns_file, "r") as f:
                for line in f.readlines():
                    self.non_freq_nouns.append(line.strip())

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

    def jaccard_matrix(self, titles, texts, diag_val=0):
        processed_nouns = []
        for i in range(len(titles)):
            processed_nouns.append(set(titles[i]).union(set(texts[i])))

        sim_matrix = [[0] * len(titles) for i in range(len(titles))]

        for i in range(len(titles)):
            for j in range(i, len(titles)):
                if i == j:
                    sim_matrix[i][j] = 1
                else:
                    union = processed_nouns[i].union(processed_nouns[j])
                    inter = processed_nouns[i].intersection(processed_nouns[j])

                    if len(union) == 0:
                        sim_matrix[i][j] = sim_matrix[j][i] = 0
                    else:
                        sim_matrix[i][j] = sim_matrix[j][i] = len(inter) / len(union)

        return sim_matrix

    def title_cosine_matrix(self, titles, texts, diag_val=0):
        titles_w2v_sum = self.compute_w2v_avg(titles)
        sim_matrix = cdist(titles_w2v_sum, titles_w2v_sum, metric="cosine")

        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix[i])):
                if (
                    np.isnan(sim_matrix[i][j])
                    or sim_matrix[i][j] < 0
                    or sim_matrix[i][j] > 1
                ):
                    sim_matrix[i][j] = 0

        return sim_matrix

    def all_cosine_matrix(self, titles, texts):
        titles_w2v_sum = self.compute_w2v_avg(titles)
        sim_matrix_titles = cdist(titles_w2v_sum, titles_w2v_sum, metric="cosine")

        texts_w2v_sum = self.compute_w2v_avg(texts)
        sim_matrix_texts = cdist(texts_w2v_sum, texts_w2v_sum, metric="cosine")

        sim_matrix = sim_matrix_titles
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix[i])):
                sim_matrix[i][j] += sim_matrix_texts[i][j]
                sim_matrix[i][j] /= 2

        return sim_matrix

    def compute_similarity_matrix(self, titles, texts, sim, diag_val):
        if sim == "jaccard":
            sim_matrix = self.jaccard_matrix(titles, texts)
        elif sim == "title_cosine":
            sim_matrix = self.title_cosine_matrix(titles, texts)
        elif sim == "all_cosine":
            return self.all_cosine_matrix(titles, texts)

        for i in range(len(sim_matrix)):
            sim_matrix[i][i] += diag_val
        print("Sim matrix computed")

        return sim_matrix

    def clusterize_affinity_propagation(self, titles, texts, sim, diag_val):
        processed_titles, processed_texts = self.process_title_text(titles, texts)

        sim_matrix = self.compute_similarity_matrix(
            processed_titles, processed_texts, sim, diag_val
        )
        af_prop = AffinityPropagation(affinity="precomputed")
        af = af_prop.fit(sim_matrix)
        print("Afinity done")
        return af, sim_matrix

    def text_to_vector(self, titles, texts, vector_repr):
        if vector_repr == "w2v_titles":
            return self.compute_w2v_avg(titles)
        elif vector_repr == "w2v_texts":
            return self.compute_w2v_avg(texts)
        elif vector_repr == "w2v_titles_texts":
            vector_titles = self.compute_w2v_avg(titles)
            vector_texts = self.compute_w2v_avg(texts)
            return [sum(e) / len(e) for e in zip([vector_titles, vector_texts])]
        elif vector_repr == "bert":
            input_pairs = [x for x in zip([titles, texts])]
            model = keras.models.load_model("models/bert-cls-256")
            bert = BertWrapper(Lang.RO, max_seq_len=128)
            input_feed = bert.process_input(input_pairs)
            predictions = model.predict(input_feed, batch_size=32)
            print(predictions)

    def clusterize_birch(self, titles, texts, vector_repr):
        processed_titles, processed_texts = self.process_title_text(titles, texts)

        vector_representation = self.text_to_vector(
            processed_titles, processed_texts, vector_repr
        )
        brc = Birch().fit(vector_representation)
        predictions = brc.predict(vector_representation)
        print(brc.labels_)
        print(brc.subcluster_centers_)
        print(brc.subcluster_labels_)

        return predictions

    def clusterize_dbscan(self, titles, texts, vector_repr):
        processed_titles, processed_texts = self.process_title_text(titles, texts)

        vector_represenation = self.text_to_vector(
            processed_titles, processed_texts, vector_repr
        )
        dbscan = DBSCAN().fit(np.array(vector_represenation))
        predictions = dbscan.labels_

        return predictions
