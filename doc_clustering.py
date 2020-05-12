import numpy as np

from sklearn.cluster import AffinityPropagation, Birch, DBSCAN

from scipy.spatial.distance import cdist

from rb.core.lang import Lang
from rb.parser.spacy_parser import SpacyParser

import helpers

class DocClustering:
    MAX_NB_WORDS = 1000000
    ENGLISH = "en"

    nlp_ro = SpacyParser.get_instance().get_model(Lang.RO)

    word_embeddings_model = "word2vec.model"
    embeddings_dim = 300

    def __init__(self, w2v_needed=False, non_freq_nouns=False):

        if w2v_needed:
            self.word_emb = helpers.load_w2v_model(self.word_embeddings_model)

        if non_freq_nouns:
            self.non_freq_nouns = []
            with open(self.nouns_file, "r") as f:
                for line in f.readlines():
                    self.non_freq_nouns.append(line.strip())

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

    def jaccard_matrix(self, titles, texts):
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

    def cosine_matrix(self, vectors):
        sim_matrix = cdist(vectors, vectors, metric="cosine")

        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix[i])):
                if (
                    np.isnan(sim_matrix[i][j])
                    or sim_matrix[i][j] < 0
                    or sim_matrix[i][j] > 1
                ):
                    sim_matrix[i][j] = 0

        return sim_matrix

    def compute_similarity_matrix(self, sim, diag_val, titles, texts, vectors):
        if sim == "jaccard":
            processed_titles, processed_texts = self.process_title_text(titles, texts)
            sim_matrix = self.jaccard_matrix(processed_titles, processed_texts)
        elif sim == "cosine":
            sim_matrix = self.cosine_matrix(vectors)

        for i in range(len(sim_matrix)):
            sim_matrix[i][i] += diag_val
        print("Sim matrix computed")

        return sim_matrix

    def clusterize_affinity_propagation(self, sim, diag_val, titles=None, texts=None, vectors=None):
        sim_matrix = self.compute_similarity_matrix(
            sim, diag_val, titles, texts, vectors
        )
        af_prop = AffinityPropagation(affinity="precomputed")
        af = af_prop.fit(sim_matrix)
        print("Afinity done")
        return af, sim_matrix

    def clusterize_birch(self, vectors):
        brc = Birch().fit(vectors)
        print('Fit ready')
        predictions = brc.predict(vectors)
        print('Predict ready')

        return predictions

    def clusterize_dbscan(self, vectors):
        dbscan = DBSCAN()
        predictions = dbscan.fit_predict(np.array(vectors))
        print('Fit predict ready')

        return predictions

