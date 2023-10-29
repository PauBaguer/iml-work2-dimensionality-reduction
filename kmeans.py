import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import davies_bouldin_score, silhouette_score

#####################################
#         Metric definitions        #
#####################################


# Compute the Manhattan distance between each point in point_vector and the variable point.
def l1_distance(point, point_vector):
    return np.sqrt(np.sum(np.absolute(point - point_vector), axis=1))


# Compute the Euclidean distance between each point in point_vector and the variable point.
def l2_distance(point, point_vector):
    return np.sqrt(np.sum((point - point_vector) ** 2, axis=1))


# Compute the cosine similarity distance between each point in point_vector and the variable point.
def cosine_distance(point, point_vector):
    point = np.array(point)
    point_vector = np.array(point_vector)
    dot_product = np.dot(point_vector, point)
    norm_x = np.sqrt(np.sum(point ** 2))
    norm_y = np.sqrt(np.sum(point_vector ** 2, axis=1))
    return 1 - dot_product / (norm_x * norm_y)


metrics = {
    'l1': l1_distance,
    'l2': l2_distance,
    'cosine': cosine_distance,
}


#####################################
#           Initializations         #
#####################################

# Random initialization
def random_init(data, num_clusters, metric):
    return data[np.random.choice(data.shape[0], num_clusters, replace=False)]


# Use kmeans++ for the initialization
def kmeanspp_init(data, num_clusters, metric):
    # Choose one center uniformly at random
    centroids = [random.choice(data)]
    for _ in range(num_clusters - 1):
        # Compute the distances from points to the centroids
        prob = np.min([metric(c, data) ** 2 for c in centroids], axis=0)
        # Normalize the vector of probabilities
        prob /= np.sum(prob)
        # Choose another centroid based on the probabilities stored in prob
        index, = np.random.choice(range(len(data)), size=1, p=prob)
        centroids += [data[index]]
    return centroids


initializations = {
    'random': random_init,
    'kmeans++': kmeanspp_init,
}

#####################################
#              Clustering           #
#####################################


def kmeans(data, num_clusters, max_iter=5000, metric='l2', centroid_init='kmeans++'):
    # Choose the metric to be used
    metric_function = metrics[metric]
    init_function = initializations[centroid_init]

    # Choose the initial centroids
    centroids = init_function(data, num_clusters, metric_function)

    i = 0
    prev_centroids = None
    # Iterate until the clusters stabilize, or we reach the maximum of iterations
    while np.not_equal(centroids, prev_centroids).any() and i < max_iter:
        clusters = [[] for _ in range(num_clusters)]
        # Assign each point to the cluster with the closest centroid
        for row in data:
            distance = [metric_function(row, centroids)]
            index = np.argmin(distance)
            clusters[index].append(row)

        # Assign the used centroids to the previous ones
        prev_centroids = centroids
        # Recompute the new centroids
        centroids = [np.mean(cpoints, axis=0) for cpoints in clusters]
        # Handle empty clusters
        # It could be improved by choosing the centroid from the cluster with highest SSE
        # or choosing the point that is farthest away from any current centroid
        for i in range(num_clusters):
            if np.isnan(centroids[i]).any():
                centroids[i] = prev_centroids[i]
        i += 1
    # Compute the final cluster assignation for the data
    assigned_cluster = []
    for row in data:
        distance = [metric_function(row, centroids)]
        index = np.argmin(distance)
        assigned_cluster.append(index)
    return centroids, assigned_cluster

def plot_kmeans_graphs(df, dataset_name, real_k, range_k=4):
    n_clusters = []
    db_scores = []
    sil_scores = []

    db_scores2 = []
    sil_scores2 = []

    db_scores3 = []
    sil_scores3 = []

    kmin = max(real_k - range_k, 2)
    kmax = real_k + range_k
    for k in range(kmin, kmax):
        n_clusters.append(k)

        centroid, labels = kmeans(df, k, metric='l1', centroid_init='kmeans++')
        db = davies_bouldin_score(df, labels)
        sil = silhouette_score(df, labels)
        db_scores.append(db)
        sil_scores.append(sil)

        centroid, labels = kmeans(df, k, metric='l2', centroid_init='kmeans++')
        db = davies_bouldin_score(df, labels)
        sil = silhouette_score(df, labels)
        db_scores2.append(db)
        sil_scores2.append(sil)

        centroid, labels = kmeans(df, k, metric='cosine', centroid_init='kmeans++')
        db = davies_bouldin_score(df, labels)
        sil = silhouette_score(df, labels)
        db_scores3.append(db)
        sil_scores3.append(sil)

    results_df = pd.DataFrame({"num_clusters": n_clusters, "DB_l1": db_scores, "SC_l1": sil_scores,
                               "DB_l2": db_scores2, "SC_l2": sil_scores2, "DB_cos": db_scores3, "SC_cos": sil_scores3})

    fig, ax = plt.subplots()
    ax.plot(results_df["num_clusters"], results_df["DB_l1"], color='r', label='Manhattan distance')
    ax.plot(results_df["num_clusters"], results_df["DB_l2"], color='g', label='Euclidean distance')
    ax.plot(results_df["num_clusters"], results_df["DB_cos"], color='b', label='Cosine dissimilarity')
    ax.legend()
    ax.set_title(f"K-Means Davies-Bouldin, {dataset_name} dataset")
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Davies-Bouldin')
    fig.savefig(f"figures/kmeans/{dataset_name}-kmeans-db.png")
    # fig.show()

    fig, ax = plt.subplots()
    ax.plot(results_df["num_clusters"], results_df["SC_l1"], color='r', label='Manhattan distance')
    ax.plot(results_df["num_clusters"], results_df["SC_l2"], color='g', label='Euclidean distance')
    ax.plot(results_df["num_clusters"], results_df["SC_cos"], color='b', label='Cosine dissimilarity')
    ax.legend()
    ax.set_title(f"K-Means Silhouette, {dataset_name} dataset")
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette')
    fig.savefig(f"figures/kmeans/{dataset_name}-kmeans-silhouette.png")
    # fig.show()

def plot_clusters(data, centroids, assigned_cluster, true_labels):
    sns.scatterplot(x=[x[0] for x in data],
                    y=[x[1] for x in data],
                    hue=true_labels,
                    style=assigned_cluster,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in centroids],
             [y for _, y in centroids],
             '+',
             markersize=10,
             )
    plt.show()
