from scipy.io import arff
import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from pca import Pca
import birch, kmeans
from visualization import Visualization
import reduceDimensionality

def load_arff(f_name):
    print(f'Opening, {f_name}')
    data, meta = arff.loadarff(f_name)
    df = pd.DataFrame(data)
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #####################################
    #             Load datasets         #
    #####################################
    adult_df = load_arff('datasets/adult.arff')
    vowel_df = load_arff('datasets/vowel.arff')
    pen_based_df = load_arff('datasets/pen-based.arff')

    print(adult_df.head(5))
    print(vowel_df.head(5))
    print(pen_based_df.head(5))


    #####################################
    #             Preprocessing         #
    #####################################

    preprocessing = Preprocessing(adult_df, vowel_df, pen_based_df, plot=False)

    print(f"preprocessed_adult_df_dimensionality: {preprocessing.pp_adult_df.shape}")
    print(f"preprocessed_vowel_df_dimensionality: {preprocessing.pp_vowel_df.shape}")
    print(f"preprocessed_pen_df_dimensionality: {preprocessing.pp_pen_based_df.shape}")
    print()


    #####################################
    #                PCA                #
    #####################################

    # If k = -1, take only eigenvectors where eigenvalues are > 1.
    adult_pca = Pca(preprocessing.pp_adult_df, dataset_name='Adult', k=-1)
    vowel_pca = Pca(preprocessing.pp_vowel_df, dataset_name='Vowel', k=-1)
    pen_based_pca = Pca(preprocessing.pp_pen_based_df, dataset_name='Pen based', k=-1)

    # # If k = -1, take only eigenvectors where eigenvalues are > 1.
    # test_pca = Pca(test_arr2, dataset_name='test', k=-1)
    
    # labels_adult_svd_birch,_ = Cluster('adult', 2, preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, 'svd').clustering('birch', 2)
    # labels_adult_svd_kmeans,_ = Cluster('adult', 2, preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, 'svd').clustering('kmeans', 2)
    #
    # labels_adult_pca_birch,_ = Cluster('adult', 2, preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, 'pca').clustering('birch', 2)
    # labels_adult_pca_kmeans,_ = Cluster('adult', 2, preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, 'pca').clustering('kmeans', 2)
    #
    #
    #####################################
    #             Truncated SVD         #
    #####################################
    cluster = reduceDimensionality.Cluster("Pen-Based", 10, preprocessing.pp_pen_based_df, preprocessing.pp_gs_pen_based_df, 'truncatedSVD')
    cluster.plot_total_explained_variance()
    n1 = 7
    n2 = 15
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='NMI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='NMI')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='kmeans')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='birch')

    cluster = reduceDimensionality.Cluster("Vowel", 11, preprocessing.pp_vowel_df, preprocessing.pp_gs_vowel_df, 'truncatedSVD')
    cluster.plot_total_explained_variance()
    n1 = 10
    n2 = 24
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='NMI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='NMI')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='kmeans')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='birch')

    cluster = reduceDimensionality.Cluster("Adult", 2, preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, 'truncatedSVD')
    cluster.plot_total_explained_variance()
    n1 = 30
    n2 = 45
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='NMI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='NMI')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='kmeans')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='birch')


    #####################################
    #               Own PCA             #
    #####################################

    cluster = reduceDimensionality.Cluster("Pen-Based", 10, preprocessing.pp_pen_based_df, preprocessing.pp_gs_pen_based_df, 'PCA')
    cluster.plot_total_explained_variance()
    n1 = 7
    n2 = 15
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='NMI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='NMI')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='kmeans')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='birch')

    cluster = reduceDimensionality.Cluster("Vowel", 11, preprocessing.pp_vowel_df, preprocessing.pp_gs_vowel_df, 'PCA')
    cluster.plot_total_explained_variance()
    n1 = 10
    n2 = 25
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='NMI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='NMI')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='kmeans')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='birch')

    cluster = reduceDimensionality.Cluster("Adult", 2, preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, 'PCA')
    cluster.plot_total_explained_variance()
    n1 = 30
    n2 = 45
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='kmeans', external_index='NMI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='Purity')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='ARI')
    cluster.plot_external_index(n_min=n1, n_max=n2, c_algorithm='birch', external_index='NMI')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='kmeans')
    cluster.plot_clustering(n_min=n1, n_max=n2, range_k=1, c_algorithm='birch')

    ############################
    #      VISUALIZATION       #
    ############################

    # adult
    adult_birch_labels, adult_birch = birch.birch(preprocessing.pp_adult_df, 2, 0.5)
    # end_time_b = time.time()
    # print(f"Training time Adult Birch: {end_time_b - start_time_b} seconds")
    # start_time_k = time.time()
    centroid_adult, adult_kmeans_labels = kmeans.kmeans(preprocessing.pp_adult_df, 2)
    # end_time_k = time.time()
    # print(f"Training time Adult Kmeans: {end_time_k - start_time_k} seconds")
    vis_adult = Visualization(preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, adult_birch_labels,
                              adult_kmeans_labels)
    # vis_adult.func_pca(dataset='Adult', n_components=2)
    vis_adult.func_isomap(dataset='Adult', n_neighbors=200, n_components=2)

