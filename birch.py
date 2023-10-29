import numpy as np
from sklearn import metrics
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def birch(X, threshold, n_clusters):
    brc = Birch(n_clusters=n_clusters, threshold=threshold).fit(X)
    res = brc.predict(X)
    labels = brc.labels_
    return labels, brc


def plot_data(X, labels, dataset_name):

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)


    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)


    print("Estimated number of points per cluster")
    points_per_cluster = {}
    unique_labels = set(labels)

    for l in unique_labels:
        if l == -1:
            points_per_cluster[l] = list(labels).count(l)
            print(f"Noise points, {l}: {points_per_cluster[l]} points")
            break
        points_per_cluster[l] = list(labels).count(l)
        print(f"Cluster {l}: {points_per_cluster[l]} points")


    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    i=0
    for row in X:
        if labels[i] == -1:
            plt.plot(
                row[0],
                row[1],
                ".",
                color='k',
                markersize=2,
                zorder=-1
            )
        else:
            plt.plot(
                row[0],
                row[1],
                ".",
                color=colors[labels[i]],
                markersize=3,
                zorder=labels[i]
            )
        i = i+1

    plt.title(f"BIRCH {dataset_name}, Nº clusters: {n_clusters_}")
    plt.savefig(f"figures/birch/{dataset_name}-graph.png")
    plt.show()

def accuracy(gs, labels):
    count = 0
    for idx, x in enumerate(gs):
        if x == labels[idx]:
            count = count + 1

    acc = count / len(gs)
    print(f'Count {count}')
    print(f'Accuracy {acc}')