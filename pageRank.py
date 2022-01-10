import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix


def build_graph(filename):
    if ".txt" in filename:
        skip_rows = 0
        with open(filename, "r") as file:
            for line in file:
                if line[0] == "#":
                    skip_rows += 1
        df = pd.read_csv(filename, header=None, sep="\t", skiprows=skip_rows)

        step = df.shape[1] // 2

        col_1 = df.iloc[:, 0].values
        col_2 = df.iloc[:, step].values
    else:
        df = pd.read_csv(filename, header=None, sep="\n")
        df = df[0].str.split(",", expand=True)

        step = df.shape[1] // 2

        col_2 = df.iloc[:, 0].values
        col_1 = df.iloc[:, step].values

        if isinstance(col_1[0], str):
            col_1 = np.array([re.sub(r"[^a-zA-Z0-9\s]", "", val.strip()) for val in col_1])

    graph = {}
    for i in range(col_1.size):
        if col_1[i] not in graph:
            graph[col_1[i]] = [col_2[i]]
        else:
            graph[col_1[i]].append(col_2[i])
        if col_2[i] not in graph:
            graph[col_2[i]] = []

    return graph


def graph(filename):
    graph = build_graph(filename)

    keys = list(graph.keys())
    node_count = len(keys)  # |V|, constant
    key_index = {keys[i]: i for i in range(node_count)}
    ranks = np.array([1 / node_count] * node_count)  # R_0
    connections = lil_matrix((node_count, node_count))
    count = 0
    for key in keys:
        for connection in graph[key]:
            connections[key_index[key], key_index[connection]] = 1
            count += 1
    print(count)

    connections = connections.tocsr()

    return node_count, ranks, connections, keys


def page_rank(filename, d, threshold=0.001, iters=15, iterations=False):
    start = time.time()
    node_count, old_ranks, connections, keys = graph(filename)

    if ".txt" not in filename:
        sums = connections.sum(axis=1)
        connections = connections.tolil()
        for i in range(len(sums)):
            if sums[i] == 0:
                connections[i] = 1  # fill sink nodes
    connections = connections.multiply(1 / connections.sum(axis=1))

    print("Preprocessing time (seconds): %.4f" % (time.time() - start))
    start = time.time()
    new_ranks = (1 - d) / node_count + d * (lil_matrix(old_ranks).dot(connections).toarray().flatten())

    iteration_count = 0
    if iterations:
        for _ in range(iters):
            iteration_count += 1
            old_ranks = new_ranks.copy()
            new_ranks = (1 - d) / node_count + d * (lil_matrix(old_ranks).dot(connections).toarray().flatten())
    else:
        while np.absolute(new_ranks - old_ranks).sum() >= threshold:
            iteration_count += 1
            old_ranks = new_ranks.copy()
            new_ranks = (1 - d) / node_count + d * (lil_matrix(old_ranks).dot(connections).toarray().flatten())

    print("Ranking time (seconds): %.4f" % (time.time() - start))
    print("# of Iterations:", iteration_count)

    return new_ranks, keys


def graph_graph(filename):
    import networkx as nx
    import matplotlib.pyplot as plt

    node_count, ranks, connections, keys = graph(filename)

    G = nx.from_numpy_matrix(connections.toarray())
    nx.draw_networkx(G)
    plt.title("karate.csv Connections Graph")
    plt.show()


if __name__ == "__main__":
    fn = sys.argv[1]
    d = float(sys.argv[2])
    if sys.argv[3] == "1":
        iterations = int(sys.argv[4])
        ranks, names = page_rank(fn, d, iters=iterations, iterations=True)
    else:
        thresh = float(sys.argv[4])
        ranks, names = page_rank(fn, d, thresh)
        print(ranks.shape)

    ranks = sorted(list(zip(ranks, names)), key=lambda x: x[0], reverse=True)
    print("\n=========== Top 10 Rankings ===========")
    for i, (p_r, name) in enumerate(ranks[:10]):
        print("%d - %s with pagerank: %.10f" % (i + 1, name, p_r))
    #
    # with open(fn + "_ranks.csv", "w") as file:
    #     file.write("ranking,name\n")
    #     for p_r, name in ranks:
    #         file.write(str(p_r) + "," + str(name) + "\n")
    # print("\nRanking successfully written to:", fn + "_ranks.csv")

    # import matplotlib.pyplot as plt
    # # 0.0468, 0.000, 0.000, 0.0219, 0.1250, 0.3371, 11.1448
    # # 13, 10, 17, 13, 7, 12, 18
    # # karate, dolphin, lesmis, football, p2p, wiki, amazon
    # # plt.plot([156, 318, 508, 1537, 31839, 103689, 3356824], [0.0468, 0.000, 0.000, 0.0219, 0.1250, 0.3371, 11.1448])
    # # karate, dolphin, lesmis, football, wiki, p2p, amazon
    # plt.plot([34, 62, 77, 340, 7115, 8846, 410236], [0.0468, 0.000, 0.000, 0.0219, 0.3371, 0.1250, 11.1448])
    # plt.xscale('log')
    # plt.xlabel("# of Nodes")
    # plt.ylabel("Time to Construct Graph (seconds)")
    # plt.title("Amount of time it takes to make the graph based on # of Nodes")
    # plt.show()
