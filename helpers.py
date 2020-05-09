import pprint
import io
import numpy as np
import csv
import sys
import json
import heapq

from functools import reduce
from operator import itemgetter 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import normalize

from scipy.spatial.distance import cdist

from gensim.models.keyedvectors import KeyedVectors


sys.path.insert(0, "readerbenchpy")

from rb.parser.spacy_parser import SpacyParser
from rb.core import lang

from collections import Counter

class DocClustering:
    MAX_NB_WORDS = 1000000
    ENGLISH = 'en'
    
    nlp_ro = SpacyParser.get_instance().get_model(lang.Lang.RO)
    
    word_embeddings_model = 'word2vec.model'
    embeddings_dim = 300

    def __init__(self):
        
        if args.title_cosine:
            self.load_w2v_model()

        if args.non_freq_nouns:
            self.non_freq_nouns = []
            with open(self.nouns_file, 'r') as f:
                for line in f.readlines():
                    self.non_freq_nouns.append(line.strip())

    
    def load_w2v_model(self):
        print('Loading model')
        self.word_emb = KeyedVectors.load_word2vec_format(
                            self.word_embeddings_model, binary=False)
        print('Model loaded')


    def load_data(self, filename='romanian_news.csv'):
        global args
        global sources_dict
        
        csv.field_size_limit(10000000)

        texts, titles, sources, sources_numbers = [], [], [], []
        cnt_skipped_examples = 0

        used_sources = {}
        
        input_csv = self.input_file

        with io.open(input_csv, 'r', encoding='utf-8', errors='replace') as csv_file:
            index = 0
            csv_reader = csv.reader(csv_file, delimiter = ',')

            for line in csv_reader:
                index += 1
                # skip first x examples
                if index > 1:
                    try:
                        title = line[1].strip()
                        source = line[0].strip()

                    except:
                        cnt_skipped_examples += 1
                        continue
                    
                    txt = line[2]
                    
                    if source not in used_sources:
                        sources_dict[len(sources_dict)] = source
                        used_sources[source] = len(sources_dict) - 1

                    titles.append(title)
                    texts.append(txt)
                    sources.append(source)
                    sources_numbers.append(used_sources[source])

            print('dataset loaded, skipped examples {} from total of {}, remaining {}'.format(cnt_skipped_examples, index, len(sources)))
            print('labels distribution, all: ')
            counters_sources = Counter(sources)
            print(counters_sources)

        if args.small_run == True:
            sources_numbers = sources_numbers[:100]
            titles = titles[:100]
            texts = texts[:100]

        self.sources_numbers = sources_numbers
        self.titles = titles
        self.texts = texts
        
        return titles, texts
    
    
    def preprocess_text(self, texts):
        global args

        docs = [self.nlp_ro(text) for text in texts]
        docs = [[token for token in doc if not token.is_stop] for doc in docs]
        
        if args.nouns_only:
            docs = list(map(lambda doc: list(filter(lambda token: token.tag_.startswith('N'), 
                    doc)), docs))
        elif args.non_freq_nouns:
            docs = list(map(lambda doc: list(filter(lambda token: token.tag_.startswith('Np') or str(token.lemma_).lower().strip() in self.non_freq_nouns, 
                    doc)), docs))
            #docs = list(map(lambda doc: list(filter(lambda token: str(token.lemma_).lower().strip() in self.non_freq_nouns, doc)), docs))
        else:
            docs = list(map(lambda doc: list(filter(lambda token: not token.tag_.startswith('P'), 
                    doc)), docs))
            
        for i, doc in enumerate(docs):
            docs[i] = [token.lemma_.lower() for token in doc]

        #docs = list(map(lambda doc: " ".join([token.lemma_.lower().strip() for token in doc]), docs))
    
        return docs
    
    
    
    
    def jaccard_matrix(self, texts, titles):
        global args

        processed_titles = self.preprocess_text(titles)
        print('Titles processed')
        processed_texts = self.preprocess_text(texts)
        print('Texts processed')
        
        processed_nouns = []
        for i in range(len(processed_titles)):
            processed_nouns.append(set(processed_titles[i]).union(set(processed_texts[i])))

        sim_matrix = [[0] * len(processed_titles) for i in range(len(processed_titles))]

        for i in range(len(processed_titles)):
            for j in range(i, len(processed_titles)):
                if i == j:
                    sim_matrix[i][j] = 1
                else:
                    union = processed_nouns[i].union(processed_nouns[j])
                    inter = processed_nouns[i].intersection(processed_nouns[j])

                    if len(union) == 0:
                        sim_matrix[i][j] = sim_matrix[j][i] = 0
                    else:
                        sim_matrix[i][j] = sim_matrix[j][i] = len(inter) / len(union)


        for i in range(len(sim_matrix)):
            sim_matrix[i][i] += args.diag_val
            
        print('Similarity matrix computed')
        return sim_matrix


    def title_cosine_matrix(self, texts, titles):
        global args
        
        processed_titles = self.preprocess_text(titles)
        print('Titles processed')
        processed_texts = self.preprocess_text(texts)
        print('Texts processed')

        titles_w2v_sum = doc_classifier.computer_word2vec_avg(processed_titles)
        
        sim_matrix = cdist(titles_w2v_sum, titles_w2v_sum, metric='cosine')
        
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix[i])):
                if np.isnan(sim_matrix[i][j]) or sim_matrix[i][j] < 0 or sim_matrix[i][j] > 1:
                    sim_matrix[i][j] = 0
        

        for i in range(len(sim_matrix)):
            sim_matrix[i][i] += args.diag_val

        print('Similarity matrix computed')
        return sim_matrix
        

    def compute_similarity_matrix(self, texts, titles):
        global args

        if args.jaccard:
            return self.jaccard_matrix(texts, titles)
        elif args.title_cosine:
            return self.title_cosine_matrix(texts, titles)

        
    def cluster_articles(self, similarity_matrix):
        afprop = AffinityPropagation(affinity='precomputed')
        af = afprop.fit(similarity_matrix)
        print('Afinity done')
        self.write_results(af, similarity_matrix)
        
    
