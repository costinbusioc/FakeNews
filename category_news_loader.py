import pprint
import io
import csv
import json
import heapq
import docclustering

import argparse

args = None
sources_dict = {}

from collections import Counter


class DatasetLoader:
    input_file = "category_news.csv"
    output_file = "results.txt"
    nouns_file = "non_freq_nouns.txt"

    def __init__(self):
        self.clusterizer = docclustering.DocClustering()

    def load_data(self):
        global args
        global sources_dict

        csv.field_size_limit(10000000)

        texts, titles, categories = [], [], []
        cnt_skipped_examples = 0

        input_csv = self.input_file

        with io.open(input_csv, "r", encoding="utf-8", errors="replace") as csv_file:
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
            print("labels distribution, all: ")
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

    def write_affinity_results(self, af, sim):
        global args

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

            cluster_data = {
                "idx": i,
                "center": self.titles[cluster_centers[i]],
                "members": cluster_titles,
                "len": len(cluster_titles),
                "categories": cluster_categories,
                "distances": similarities,
                "closest": closest[i],
                "farest": farest[i],
            }

            result[i] = cluster_data

        # with open('json_format.json', 'w') as fp:
        # json.dump(result, fp, indent=4)

        with io.open(args.out_file, "w", encoding="utf-8") as outputfile:
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

    def cluster_dataset(self):
        global args
        titles, texts = dataset_loader.load_data()

        if args.clust_alg == "affinity":
            af, sim_matrix = self.clusterizer.clusterize_affinity_propagation(
                titles, texts, args.affinity_sim, args.diag_val
            )
            self.write_affinity_results(af, sim_matrix)
        elif args.clust_alg == "birch":
            predictions = self.clusterizer.clusterize_birch(
                titles, texts, args.vector_repr
            )
            print(predictions)


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
        default="affinity",
        choices=["affinity", "birch"],
        help="Clusterization algorithm use",
    )
    parser.add_argument(
        "--affinity_sim",
        dest="affinity_sim",
        action="store",
        type=str,
        default="jaccard",
        choices=["jaccard", "title_cosine", "all_cosine"],
        help="Similarity to be used by affinity for clusterization",
    )
    parser.add_argument(
        "--vector_repr",
        dest="vector_repr",
        action="store",
        type=str,
        default="w2v_titles",
        choices=["w2v_titles", "w2v_texts", "w2v_titles_texts"],
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
    dataset_loader.cluster_dataset()
