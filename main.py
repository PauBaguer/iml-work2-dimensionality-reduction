from scipy.io import arff
import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from pca import Pca
import birch, kmeans
from visualization import Visualization
import time
from sklearn_PCA import SklearnAlgorithms
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


    ############################
    #   SKLEARN PCA AND IPCA   #
    ############################

    sk_adult = SklearnAlgorithms(preprocessing.pp_adult_df)
    sk_pen = SklearnAlgorithms(preprocessing.pp_pen_based_df)
    sk_vowel = SklearnAlgorithms(preprocessing.pp_vowel_df)

    # 1. NUMBER OF COMPONENTS STUDY

    # adult-based
    feat_adult = preprocessing.pp_adult_df.shape[1]  # 108
    n_components_adult = np.arange(1, feat_adult)

    # pen-based
    feat_pen = preprocessing.pp_pen_based_df.shape[1]  # 16
    n_components_pen = np.arange(1, feat_pen)

    # vowel
    feat_vowel = preprocessing.pp_vowel_df.shape[1]  # 29
    n_components_vowel = np.arange(1, feat_vowel)

    # # 1.1. PCA

    # adult
    variances_pca_adult, thresh_adult = sk_adult.explore_ncomponents(n_components_adult, True, 'Adult')
    start_time = time.time()
    results_pca_adult = sk_adult.func_sklearn(0, thresh_adult, True, 'Adult')
    end_time = time.time()
    print(f"PCA time Adult: {end_time - start_time} seconds")

    # pen-based
    variances_pca_pen, thresh_pen = sk_pen.explore_ncomponents(n_components_pen, True, 'Pen-based')
    start_time = time.time()
    results_pca_pen = sk_pen.func_sklearn(0, thresh_pen, True, 'Pen-based')  # CHANGE n_components
    end_time = time.time()
    print(f"PCA time Pen-based: {end_time - start_time} seconds")

    # vowel
    variances_pca_vowel, thresh_vowel = sk_vowel.explore_ncomponents(n_components_vowel, True, 'Vowel')
    start_time = time.time()
    results_pca_vowel = sk_vowel.func_sklearn(0, thresh_vowel, True, 'Vowel')
    end_time = time.time()
    print(f"PCA time Vowel: {end_time - start_time} seconds")

    # # 1.2. IPCA

    # # adult-based
    variances_ipca_adult = sk_adult.explore_ncomponents(n_components_adult, True, 'Adult', 1)
    start_time = time.time()
    results_ipca_adult = sk_adult.func_sklearn(1, 23, True, 'Adult')
    end_time = time.time()
    print(f"IPCA time Adult: {end_time - start_time} seconds")

    # pen-based
    variances_ipca_pen = sk_pen.explore_ncomponents(n_components_pen, True, 'Pen-based', 1)
    start_time = time.time()
    results_ipca_pen = sk_pen.func_sklearn(1, 8, True, 'Pen-based')
    end_time = time.time()
    print(f"IPCA time Pen-based: {end_time - start_time} seconds")

    # vowel
    variances_ipca_vowel = sk_vowel.explore_ncomponents(n_components_vowel, True, 'Vowel', 1)  # CHANGE n_components
    start_time = time.time()
    results_ipca_vowel = sk_vowel.func_sklearn(1, 10, True, 'Vowel')
    end_time = time.time()
    print(f"IPCA time Vowel: {end_time - start_time} seconds")


    # #####################################
    # #             Truncated SVD         #
    # #####################################
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
    start_time_b = time.time()
    adult_birch_labels, adult_birch = birch.birch(preprocessing.pp_adult_df, 2, 0.5)
    end_time_b = time.time()
    print(f"Training time Adult Birch: {end_time_b - start_time_b} seconds")
    start_time_k = time.time()
    centroid_adult, adult_kmeans_labels = kmeans.kmeans(preprocessing.pp_adult_df, 2)
    end_time_k = time.time()
    print(f"Training time Adult Kmeans: {end_time_k - start_time_k} seconds")

    transformed_dataset_adult = Pca(preprocessing.pp_adult_df, "Adult", 80).reduced_original_values
    start_time_b = time.time()
    adult_birch_labels_PCA, _ = birch.birch(transformed_dataset_adult, 2, 0.5)
    end_time_b = time.time()
    print(f"Training time Adult Birch: {end_time_b - start_time_b} seconds")
    start_time_k = time.time()
    _, adult_kmeans_labels_PCA = kmeans.kmeans(transformed_dataset_adult, 2)
    end_time_k = time.time()
    print(f"Training time Adult Kmeans: {end_time_k - start_time_k} seconds")
    vis_adult = Visualization(preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df, adult_birch_labels,
                              adult_kmeans_labels, adult_birch_labels_PCA, adult_kmeans_labels_PCA)
    vis_adult.func_pca(dataset='Adult', n_components=2)
    vis_adult.func_isomap_subsample(dataset='Adult')

    # pen-based
    start_time_b = time.time()
    pen_birch_labels, pen_birch = birch.birch(preprocessing.pp_pen_based_df, 10, 0.5)
    end_time_b = time.time()
    print(f"Training time Pen Birch: {end_time_b - start_time_b} seconds")
    start_time_k = time.time()
    centroid_pen, pen_kmeans_labels = kmeans.kmeans(preprocessing.pp_pen_based_df, 10)
    end_time_k = time.time()
    print(f"Training time Pen Kmeans: {end_time_k - start_time_k} seconds")

    transformed_dataset_pen = Pca(preprocessing.pp_pen_based_df, "Pen-Based", 7).reduced_original_values
    start_time_b = time.time()
    pen_birch_labels_PCA, _ = birch.birch(transformed_dataset_pen, 10, 0.5)
    end_time_b = time.time()
    print(f"Training time Pen Birch: {end_time_b - start_time_b} seconds")
    start_time_k = time.time()
    _, pen_kmeans_labels_PCA = kmeans.kmeans(transformed_dataset_pen, 10)
    vis_pen = Visualization(preprocessing.pp_pen_based_df, preprocessing.pp_gs_pen_based_df, pen_birch_labels,
                            pen_kmeans_labels, pen_birch_labels_PCA, pen_kmeans_labels_PCA)
    end_time_k = time.time()
    print(f"Training time Pen Kmeans: {end_time_k - start_time_k} seconds")
    vis_pen.func_pca(dataset='Pen-based', n_components=2)
    vis_pen.func_isomap(dataset='Pen-based', n_components=2)
    # vis_pen.func_isomap_subsample(dataset='Pen-based', n_neighbors=100, sample_size=5000, n_components=2)

    # vowel
    start_time_b = time.time()
    vowel_birch_labels, vowel_birch = birch.birch(preprocessing.pp_vowel_df, 11, 0.5)
    end_time_b = time.time()
    print(f"Training time Vowel Birch: {end_time_b - start_time_b} seconds")
    start_time_k = time.time()
    centroid_vowel, vowel_kmeans_labels = kmeans.kmeans(preprocessing.pp_vowel_df, 11)
    end_time_k = time.time()
    print(f"Training time Vowel Kmeans: {end_time_k - start_time_k} seconds")

    transformed_dataset_vowel = Pca(preprocessing.pp_vowel_df, "Vowel", 15).reduced_original_values
    start_time_b = time.time()
    vowel_birch_labels_PCA, _ = birch.birch(transformed_dataset_vowel, 11, 0.5)
    end_time_b = time.time()
    print(f"Training time Vowel Birch: {end_time_b - start_time_b} seconds")
    start_time_k = time.time()
    _, vowel_kmeans_labels_PCA = kmeans.kmeans(transformed_dataset_vowel, 11)
    end_time_k = time.time()
    print(f"Training time Vowel Kmeans: {end_time_k - start_time_k} seconds")
    vis_vowel = Visualization(preprocessing.pp_vowel_df, preprocessing.pp_gs_vowel_df, vowel_birch_labels,
                              vowel_birch_labels_PCA,vowel_kmeans_labels, vowel_kmeans_labels_PCA)
    vis_vowel.func_pca(dataset='Vowel', n_components=2)
    vis_vowel.func_isomap(dataset='Vowel', n_neighbors=100, n_components=2)

