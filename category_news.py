import pprint
import io
import numpy as np
import csv
import sys
import json
import heapq

from functools import reduce
from operator import itemgetter 

sys.path.insert(0, "rbpy")

from rb.parser.spacy_parser import SpacyParser
from rb.core import lang

import argparse
args = None
sources_dict = {}

from collections import Counter

# different doc classifiers, lstm, naive bayes, svm
class DocClassifiers:
    input_file = "category_news.csv"
    output_file = 'results.txt'
    nouns_file = 'non_freq_nouns.txt'

    nlp_ro = SpacyParser.get_instance().get_model(lang.Lang.RO)
    
    word_embeddings_model = '../word2vec.model'
    embeddings_dim = 300

    def __init__(self):
        global args
        pass        

    def load_data(self):
        global args
        global sources_dict
        
        csv.field_size_limit(10000000)

        texts, titles, categories = [], [], []
        cnt_skipped_examples = 0

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
                        text = line[2].strip()

                    except:
                        cnt_skipped_examples += 1
                        continue
                    
                    category = line[3]

                    titles.append(title)
                    texts.append(text)
                    categories.append(category)

            print('dataset loaded, skipped examples {} from total of {}, remaining {}'.format(cnt_skipped_examples, index, len(categories)))
            print('labels distribution, all: ')
            counters_categories = Counter(categories)
            print(counters_categories)

        if args.small_run == True:
            titles = titles[:100]
            texts = texts[:100]
            categories = categories[:100]

        self.categories = categories
        self.titles = titles
        self.texts = texts
        
        return titles, texts
    
    
        
    
    def all_cosine_matrix(self, processed_texts, processed_titles):
        global args
        
        titles_w2v_sum = doc_classifier.computer_word2vec_avg(processed_titles)
        sim_matrix_titles = doc_classifier.cosine_matrix(titles_w2v_sum)
        
        text_w2v_sum = doc_classifier.computer_word2vec_avg(processed_texts)
        sim_matrix_text = doc_classifier.cosine_matrix(text_w2v_sum)

        sim_matrix = sim_matrix_titles
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix[i])):
                sim_matrix[i][j] += sim_matrix_text[i][j]
                sim_matrix[i][j] /= 2

        for i in range(len(sim_matrix)):
            sim_matrix[i][i] += args.diag_val

        print('Similarity matrix computed')
        return sim_matrix

    
    def compute_similarity_matrix(self, texts, titles):
        global args
        
        processed_titles = self.preprocess_text(titles)
        print('Titles processed')
        processed_texts = self.preprocess_text(texts)
        print('Texts processed')

        if args.jaccard:
            return self.jaccard_matrix(processed_texts, processed_titles)
        elif args.title_cosine:
            return self.title_cosine_matrix(processed_texts, processed_titles)
        elif args.all_cosine:
            return self.all_cosine_matrix(processed_texts, processed_titles)
        

    def write_results(self, af, sim):
        global args

        result = {}
        
        labels = af.labels_
        cluster_centers = af.cluster_centers_indices_
        
        dist_clusters = []
        for i in range(len(cluster_centers)):
            diff = []
            for j in range(len(cluster_centers)):
                if i != j:
                    heapq.heappush(diff, (sim[cluster_centers[i]][cluster_centers[j]], j))
            dist_clusters.append(diff)

        closest = [] 
        farest = []
        for i in range(len(cluster_centers)):
            closest.append([])
            farest.append([])
            far = heapq.nsmallest(5, dist_clusters[i])
            for cls in far:
                farest[i].append((cls[1], self.titles[cluster_centers[cls[1]]]))

            close = heapq.nlargest(5, dist_clusters[i])
            for f in close:
                closest[i].append((f[1], self.titles[cluster_centers[f[1]]]))

        for i in range(len(cluster_centers)):

            indices = []
            
            for j, label in enumerate(labels):
                if label == i:
                    indices.append(j)
                    
            cluster_titles = list(map(self.titles.__getitem__, indices))
            cluster_categories = list(map(self.categories.__getitem__, indices))

            similarities = [[0] * len(indices) for i in range(len(indices))]
            for l1 in range(len(indices)):
                for l2 in range(len(indices)):
                    similarities[l1][l2] = sim[indices[l1]][indices[l2]]

            cluster_data = {
                'idx': i,
                'center': self.titles[cluster_centers[i]],
                'members': cluster_titles,
                'len': len(cluster_titles),
                'categories': cluster_categories,
                'distances': similarities,
                'closest': closest[i],
                'farest': farest[i]
            }
            
            result[i] = cluster_data
        
        #with open('json_format.json', 'w') as fp:
            #json.dump(result, fp, indent=4)
            
        with io.open(args.out_file, 'w', encoding='utf-8') as outputfile:
            for i in result:
                outputfile.write(str(i) + ':\n')
                outputfile.write(' ' * 4 + 'len: ' + str(result[i]['len']) + '\n')
                outputfile.write(' ' * 4 + 'center: ' + result[i]['center'] + '\n')
                outputfile.write(' ' * 4 + 'members: ' + '\n')
                for j in range(len(result[i]['members'])):
                    outputfile.write(' ' * 8 + '- ' + str(result[i]['categories'][j]) + ': ' + str(result[i]['members'][j]) + '\n')
                #outputfile.write(' ' * 4 + 'distances: ' + '\n')
                #for distance in result[i]['distances']:
                    #outputfile.write(' ' * 8 + str(distance) + '\n')
                
                outputfile.write(' ' * 4 + 'closest: ' + '\n')
                for member in result[i]['closest']:
                    outputfile.write(' ' * 8 + str(member[0]) + ': ' + str(member[1]) + '\n')
                #outputfile.write(' ' * 4 + 'farest: ' + '\n')
                #for member in result[i]['farest']:
                    #outputfile.write(' ' * 8 + str(member[0]) + ': ' + str(member[1]) + '\n')

                outputfile.write('\n\n')
            #pp = pprint.PrettyPrinter(indent=4, stream=outputfile, depth=4, width=200)
            #pp.pprint(result)
            #json.dump(result, outputfile, indent='\t')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--small_run', dest='small_run', action='store_true', default=False)
    parser.add_argument('--title_cosine', dest='title_cosine', action='store_true', default=False)
    parser.add_argument('--all_cosine', dest='all_cosine', action='store_true', default=False)
    parser.add_argument('--jaccard', dest='jaccard', action='store_true', default=False)
    parser.add_argument('--nouns_only', dest='nouns_only', action='store_true', default=False)
    parser.add_argument('--non_freq_nouns', dest='non_freq_nouns', action='store_true', default=False)
    parser.add_argument('--out_file', dest='out_file', action='store', type=str, default='out.txt')
    parser.add_argument('--diag_val', dest='diag_val', action='store', type=float, default=1)
    parser.add_argument('--cos_treshold', dest='cos_treshold', action='store', type=float, default=0.0)

    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])

    doc_classifier = DocClassifiers()
    all_titles, all_texts = doc_classifier.load_data()
    
    #sim_matrix = doc_classifier.compute_similarity_matrix(all_texts, all_titles)

    #doc_classifier.cluster_articles(sim_matrix)
