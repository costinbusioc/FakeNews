import pprint
import io
import csv
import json
import heapq
import random
import numpy as np
import doc_clustering
from pympler import asizeof
from operator import itemgetter
from collections import defaultdict

import argparse
from helpers import write_csv

args = None
sources_dict = {}

from collections import Counter


class DatasetLoader:
    nouns_file = "non_freq_nouns.txt"

    def __init__(self):
        global args
        self.input_file = "category_news_bert_vectors_0.2.csv"
        self.output_file = args.out_file

        w2v_needed = not args.vector_repr == "bert"
        self.clusterizer = doc_clustering.DocClustering(w2v_needed=w2v_needed)

    def load_data(self):
        global args
        global sources_dict

        csv.field_size_limit(10000000)

        texts, titles, categories, vectors = [], [], [], []
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

                        vector = line[4].strip()
                        vector = vector.split()
                        if vector[0] == "[":
                            vector = vector[1:]
                        if vector[-1][-1] == "]":
                            vector[-1] = vector[-1][:-1]
                        vector = list(map(float, vector))
                        vector = np.array(vector, dtype=np.float32)

                    except:
                        cnt_skipped_examples += 1
                        continue

                    category = line[3]

                    titles.append(title)
                    texts.append(text)
                    categories.append(category)
                    vectors.append(vector)

                    if args.small_run == True:
                        if len(vectors) == 100:
                            break

            print(
                "dataset loaded, skipped examples {} from total of {}, remaining {}".format(
                    cnt_skipped_examples, index, len(categories)
                )
            )
            print("labels distribution, all: ")
            counters_categories = Counter(categories)
            print(counters_categories)

        if args.small_run == True:
            titles = titles[:100]
            texts = texts[:100]
            categories = categories[:100]
            vectors = vectors[:100]

        self.categories = categories
        self.titles = titles
        self.texts = texts
        self.vectors = vectors
        print(f"ALL: {asizeof.asizeof(self.vectors)}")

    def keep_dataset_percent(self, percent):
        global args
        global sources_dict

        csv.field_size_limit(10000000)

        texts, titles, categories, vectors = [], [], [], []
        cnt_skipped_examples = 0

        categories_dict = defaultdict(list)

        with io.open(
            self.input_file, "r", encoding="utf-8", errors="replace"
        ) as csv_file:
            index = 0
            csv_reader = csv.reader(csv_file, delimiter=",")

            for line in csv_reader:
                # skip first x examples
                if index:
                    try:
                        title = line[1].strip()
                        text = line[2].strip()

                        vector = line[4].strip()
                        vector = vector.split()
                        if vector[0] == "[":
                            vector = vector[1:]
                        if vector[-1][-1] == "]":
                            vector[-1] = vector[-1][:-1]
                        vector = list(map(float, vector))
                        vector = np.array(vector, dtype=np.float32)

                    except:
                        cnt_skipped_examples += 1
                        continue

                    category = line[3]

                    titles.append(title)
                    texts.append(text)
                    categories.append(category)
                    vectors.append(vector)

                    categories_dict[category].append(index - 1)

                    if args.small_run == True:
                        if len(vectors) == 100:
                            break
                index += 1

            print(
                "dataset loaded, skipped examples {} from total of {}, remaining {}".format(
                    cnt_skipped_examples, index, len(categories)
                )
            )
            print("labels distribution, all: ")
            counters_categories = Counter(categories)
            print(counters_categories)

        if args.small_run == True:
            titles = titles[:100]
            texts = texts[:100]
            categories = categories[:100]
            vectors = vectors[:100]

        print(f"ALL: {asizeof.asizeof(vectors)}")

        new_texts = []
        new_titles = []
        new_categories = []
        new_vectors = []

        for category in categories_dict:
            inds = set(
                random.sample(
                    list(range(len(categories_dict[category]))),
                    int(percent * len(categories_dict[category])),
                )
            )

            new_texts += list(itemgetter(*inds)(texts))
            new_titles += list(itemgetter(*inds)(titles))
            new_categories += ([category] * len(inds))
            new_vectors += list(itemgetter(*inds)(vectors))
    
        print(Counter(new_categories))
        input_name = self.input_file.split('.')[0]
        out_file = f"{input_name}_{percent}.csv"
        write_csv(out_file, ["Title", "Text", "Category", "Vector"], [new_titles, new_texts, new_categories, new_vectors])


    def write_affinity_results(self, af, sim):
        result = {}

        labels = af.labels_
        cluster_centers = af.cluster_centers_indices_

        dist_clusters = []
        for i in range(len(cluster_centers)):
            diff = []
            for j in range(len(cluster_centers)):
                if i != j:
                    heapq.heappush(
                        diff, (sim[cluster_centers[i]][cluster_centers[j]], j)
                    )
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

            result[i] = {
                "center": self.titles[cluster_centers[i]],
                "members": cluster_titles,
                "len": len(cluster_titles),
                "categories": cluster_categories,
                "distances": similarities,
                "closest": closest[i],
                "farest": farest[i],
            }

        # with open('json_format.json', 'w') as fp:
        # json.dump(result, fp, indent=4)

        with io.open(self.output_file, "w", encoding="utf-8") as outputfile:
            for i in result:
                outputfile.write(str(i) + ":\n")
                outputfile.write(" " * 4 + "len: " + str(result[i]["len"]) + "\n")
                outputfile.write(" " * 4 + "center: " + result[i]["center"] + "\n")
                outputfile.write(" " * 4 + "members: " + "\n")
                for j in range(len(result[i]["members"])):
                    outputfile.write(
                        " " * 8
                        + "- "
                        + str(result[i]["categories"][j])
                        + ": "
                        + str(result[i]["members"][j])
                        + "\n"
                    )
                # outputfile.write(' ' * 4 + 'distances: ' + '\n')
                # for distance in result[i]['distances']:
                # outputfile.write(' ' * 8 + str(distance) + '\n')

                outputfile.write(" " * 4 + "closest: " + "\n")
                for member in result[i]["closest"]:
                    outputfile.write(
                        " " * 8 + str(member[0]) + ": " + str(member[1]) + "\n"
                    )
                # outputfile.write(' ' * 4 + 'farest: ' + '\n')
                # for member in result[i]['farest']:
                # outputfile.write(' ' * 8 + str(member[0]) + ': ' + str(member[1]) + '\n')

                outputfile.write("\n\n")
            # pp = pprint.PrettyPrinter(indent=4, stream=outputfile, depth=4, width=200)
            # pp.pprint(result)
            # json.dump(result, outputfile, indent='\t')


    def write_birch_results(self, labels):
        self.write_labels_as_csv(labels)


    def write_dbscan_results(self, labels):
        self.write_results_by_labels(labels)


    def write_labels_as_csv(self, labels):
        unique_labels = set(labels)

        all_indices = []
        all_lens = []
        for l in unique_labels:
            indices = []

            for i, label in enumerate(labels):
                if label == l:
                    indices.append(i)

            all_indices.append(indices)
            all_lens.append(len(indices))

        write_csv(self.output_file, ["Len", "Indices"], [all_lens, all_indices])


    def get_elements_by_indices(self, indices):
        cluster_titles = list(map(self.titles.__getitem__, indices))
        cluster_categories = list(map(self.categories.__getitem__, indices))

        return {
            "members": cluster_titles,
            "len": len(indices),
            "categories": cluster_categories,
        }

    def write_result(self, result):
        # with open('json_format.json', 'w') as fp:
        # json.dump(result, fp, indent=4)

        with io.open(self.output_file, "w", encoding="utf-8") as outputfile:
            for i in result:
                outputfile.write(str(i) + ":\n")
                outputfile.write(" " * 4 + "len: " + str(result[i]["len"]) + "\n")
                outputfile.write(" " * 4 + "members: " + "\n")
                for j in range(len(result[i]["members"])):
                    outputfile.write(
                        " " * 8
                        + "- "
                        + str(result[i]["categories"][j])
                        + ": "
                        + str(result[i]["members"][j])
                        + "\n"
                    )
                outputfile.write("\n\n")
            # pp = pprint.PrettyPrinter(indent=4, stream=outputfile, depth=4, width=200)
            # pp.pprint(result)
            # json.dump(result, outputfile, indent='\t')


    def write_results_by_labels(self, labels):
        result = {}
        unique_labels = set(labels)

        for l in unique_labels:
            indices = []

            for i, label in enumerate(labels):
                if label == l:
                    indices.append(i)

            result[l] = self.get_elements_by_indices(indices)

        self.write_result(result)


    def csv_to_txt(self, in_file):
        csv.field_size_limit(10000000)

        result = {}
        cnt_skipped_examples = 0

        with io.open(
                in_file, "r", encoding="utf-8", errors="replace"
        ) as csv_file:
            index = 0
            csv_reader = csv.reader(csv_file, delimiter=",")

            for line in csv_reader:
                index += 1
                # skip first x examples
                if index > 1:
                    try:
                        cluster_nr = line[0].strip()
                        cluster_values = line[2].strip()

                        indices = [int(x.strip().replace('[', '').replace(']','')) for x in cluster_values.split(',')]
                        result[cluster_nr] = self.get_elements_by_indices(indices)
                    except:
                        cnt_skipped_examples += 1
                        continue
            print(f"Could not read {cnt_skipped_examples} lines.")
        self.write_result(result)


    def cluster_dataset(self):
        if args.clust_alg == "affinity":
            af, sim_matrix = self.clusterizer.clusterize_affinity_propagation(
                args.affinity_sim, args.diag_val, self.titles, self.texts, self.vectors
            )
            self.write_affinity_results(af, sim_matrix)
        elif args.clust_alg == "birch":
            predictions = self.clusterizer.clusterize_birch(self.vectors)
            self.write_birch_results(predictions)
        elif args.clust_alg == "dbscan":
            predictions = self.clusterizer.clusterize_dbscan(self.vectors)
            self.write_dbscan_results(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        "--small_run", dest="small_run", action="store_true", default=False
    )
    parser.add_argument(
        "--clust_alg",
        dest="clust_alg",
        action="store",
        type=str,
        default="birch",
        choices=["affinity", "birch", "dbscan"],
        help="Clusterization algorithm use",
    )
    parser.add_argument(
        "--affinity_sim",
        dest="affinity_sim",
        action="store",
        type=str,
        default="jaccard",
        choices=["jaccard", "cosine"],
        help="Similarity to be used by affinity for clusterization",
    )
    parser.add_argument(
        "--vector_repr",
        dest="vector_repr",
        action="store",
        type=str,
        default="bert",
        choices=["w2v_titles", "w2v_texts", "w2v_titles_texts", "bert"],
        help="Vector text represenation for birch",
    )
    parser.add_argument(
        "--nouns_only", dest="nouns_only", action="store_true", default=False
    )
    parser.add_argument(
        "--non_freq_nouns", dest="non_freq_nouns", action="store_true", default=False
    )
    parser.add_argument(
        "--out_file", dest="out_file", action="store", type=str, default="out.txt"
    )
    parser.add_argument(
        "--diag_val",
        dest="diag_val",
        action="store",
        type=float,
        default=1,
        help="Diagonal value to be added on affinity matrix",
    )
    parser.add_argument(
        "--cos_treshold", dest="cos_treshold", action="store", type=float, default=0.0
    )

    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, "->", args.__dict__[k])

    dataset_loader = DatasetLoader()
    print(f"obj: {asizeof.asizeof(dataset_loader)}")

    dataset_loader.load_data()
    dataset_loader.csv_to_txt('results/birch/out_birch_bf6_n4.csv')
    #dataset_loader.cluster_dataset()
    #dataset_loader.keep_dataset_percent(0.2)
