from scipy.io import arff
import numpy as np
import pandas as pd
import sys
import scipy as sc
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import Preprocessing
from sklearn.decomposition import PCA
from pca import Pca
import birch, kmeans
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif

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
    #             SelectKBest           #
    #####################################
    
    # print('SelectKBest with ANOVA F-test')
    # k = 2  # Number of top features to select
    # selector_adult = SelectKBest(score_func=f_classif, k=k)
    # X_adult = selector_adult.fit_transform(preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df)
    # selected_adult = selector_adult.get_support(indices=True)
    # print(f"Selected features for adult dataset: {selected_adult}")
    #
    # selector_vowel = SelectKBest(score_func=f_classif, k=k)
    # X_vowel = selector_vowel.fit_transform(preprocessing.pp_vowel_df, preprocessing.pp_gs_vowel_df)
    # selected_vowel = selector_vowel.get_support(indices=True)
    # print(f"Selected features for vowel dataset: {selected_vowel}")
    #
    # selector_pen = SelectKBest(score_func=f_classif, k=k)
    # X_pen = selector_pen.fit_transform(preprocessing.pp_pen_based_df, preprocessing.pp_gs_pen_based_df)
    # selected_pem = selector_pen.get_support(indices=True)
    # print(f"Selected features for pen dataset: {selected_pem}")
    #
    # plt.figure(figsize=(10, 8))
    # plt.subplot(3, 1, 1)
    # plt.subplots_adjust(hspace=0.5) # Add this line to increase the distance between subplots
    # plt.scatter(X_adult[:, 0], X_adult[:, 1], c=preprocessing.pp_gs_adult_df)
    # plt.xlabel('Feature ' + str(selected_adult[0]))
    # plt.ylabel('Feature ' + str(selected_adult[1]))
    # plt.title('Feature importance (F-test) for adult dataset')
    #
    # plt.subplot(3, 1, 2)
    # plt.scatter(X_vowel[:, 0], X_vowel[:, 1], c=preprocessing.pp_gs_vowel_df)
    # plt.xlabel('Feature ' + str(selected_vowel[0]))
    # plt.ylabel('Feature ' + str(selected_vowel[1]))
    # plt.title('Feature importance (F-test) for vowel dataset')
    #
    # plt.subplot(3, 1, 3)
    # plt.scatter(X_pen[:, 0], X_pen[:, 1], c=preprocessing.pp_gs_pen_based_df)
    # plt.xlabel('Feature' + str(selected_pem[0]))
    # plt.ylabel('Feature' + str(selected_pem[1]))
    # plt.title('Feature importance (F-test) for pen dataset')
    # plt.show()
    #
    # print('\n Select k best features with chi-squared test')
    # selector_adult = SelectKBest(score_func=chi2, k=k)
    # X_adult = selector_adult.fit_transform(preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df)
    # selected_adult = selector_adult.get_support(indices=True)
    # print(f"Selected features for adult dataset: {selected_adult}")
    #
    # selector_vowel = SelectKBest(score_func=chi2, k=k)
    # X_vowel = selector_vowel.fit_transform(preprocessing.pp_vowel_df, preprocessing.pp_gs_vowel_df)
    # selected_vowel = selector_vowel.get_support(indices=True)
    # print(f"Selected features for vowel dataset: {selected_vowel}")
    #
    # selector_pen = SelectKBest(score_func=chi2, k=k)
    # X_pen = selector_pen.fit_transform(preprocessing.pp_pen_based_df, preprocessing.pp_gs_pen_based_df)
    # selected_pen = selector_pen.get_support(indices=True)
    # print(f"Selected features for pen dataset: {selected_pen}")
    #
    # plt.figure(figsize=(10, 8))
    # plt.subplots_adjust(hspace=0.5) # Add this line to increase the distance between subplots
    # plt.subplot(3, 1, 1)
    # plt.scatter(X_adult[:, 0], X_adult[:, 1], c=preprocessing.pp_gs_adult_df)
    # plt.xlabel('Feature '  + str(selected_adult[0]))
    # plt.ylabel('Feature ' + str(selected_adult[1]))
    # plt.title('Feature importance (chi-squared test) for adult dataset')
    #
    # plt.subplot(3, 1, 2)
    # plt.scatter(X_vowel[:, 0], X_vowel[:, 1], c=preprocessing.pp_gs_vowel_df)
    # plt.xlabel('Feature ' + str(selected_vowel[0]))
    # plt.ylabel('Feature ' + str(selected_vowel[1]))
    # plt.title('Feature importance (chi-squared test) for vowel dataset')
    #
    # plt.subplot(3, 1, 3)
    # plt.scatter(X_pen[:, 0], X_pen[:, 1], c=preprocessing.pp_gs_pen_based_df)
    # plt.xlabel('Feature ' + str(selected_pen[0]))
    # plt.ylabel('Feature ' + str(selected_pen[1]))
    # plt.title('Feature importance (chi-squared test) for pen dataset')
    # plt.show()
    #
    # #####################################
    # #           N best plots            #
    # #####################################
    #
    # import itertools
    #
    # # Select the top 5 features for each dataset
    # k=4
    # selector_adult = SelectKBest(score_func=f_classif, k=k)
    # X_adult = selector_adult.fit_transform(preprocessing.pp_adult_df, preprocessing.pp_gs_adult_df)
    # selected_adult = selector_adult.get_support(indices=True)
    #
    # selector_vowel = SelectKBest(score_func=f_classif, k=k)
    # X_vowel = selector_vowel.fit_transform(preprocessing.pp_vowel_df, preprocessing.pp_gs_vowel_df)
    # selected_vowel = selector_vowel.get_support(indices=True)
    #
    # selector_pen = SelectKBest(score_func=f_classif, k=k)
    # X_pen = selector_pen.fit_transform(preprocessing.pp_pen_based_df, preprocessing.pp_gs_pen_based_df)
    # selected_pen = selector_pen.get_support(indices=True)
    #
    # # Create a grid of plots for each pair of features
    # datasets = [(X_adult, preprocessing.pp_gs_adult_df, selected_adult, 'adult'),
    #             (X_vowel, preprocessing.pp_gs_vowel_df, selected_vowel, 'vowel'),
    #             (X_pen, preprocessing.pp_gs_pen_based_df, selected_pen, 'pen')]
    #
    # for X, y, selected, dataset_name in datasets:
    #     plt.figure(figsize=(10, 8))
    #     plt.subplots_adjust(hspace=0.8, wspace=0.8) # Add this line to increase the distance between subplots
    #     n=1
    #     for i, j in itertools.combinations(range(X.shape[1]), 2):
    #         plt.subplot(3, 2, n)
    #         plt.scatter(X[:, i], X[:, j], c=y)
    #         plt.xlabel('Feature ' + str(selected[i]))
    #         plt.ylabel('Feature ' + str(selected[j]))
    #         plt.title(f'{dataset_name} dataset: Features {selected[i]} vs {selected[j]}', fontsize=10)
    #         n+=1
    #     plt.show()


    #####################################
    #                PCA                #
    #####################################

    # TODO: ASK TEACHER FOR DECIMAL PROBLEM
    # TODO: add eigenvectors to plots, plot for datasets.

    # If k = -1, take only eigenvectors where eigenvalues are > 1.
    adult_pca = Pca(preprocessing.pp_adult_df, dataset_name='Adult', k=-1)
    vowel_pca = Pca(preprocessing.pp_vowel_df, dataset_name='Vowel', k=-1)
    pen_based_pca = Pca(preprocessing.pp_pen_based_df, dataset_name='Pen based', k=-1)


    test_arr2 = np.array([
        [2.5,2.4],
        [0.5,0.7],
        [2.2,2.9],
        [1.9,2.2],
        [3.1,3.0],
        [2.3,2.7],
        [2.0,1.6],
        [1.0,1.1],
        [1.5,1.6],
        [1.1,0.9]
    ])

    print("mean:")
    print(test_arr2.mean(axis=0))
    print("var:")
    print(test_arr2.var(axis=0))

    # If k = -1, take only eigenvectors where eigenvalues are > 1.
    test_pca = Pca(test_arr2, dataset_name='test', k=-1)


    # print()
    # print("SKLEARN")
    # pca = PCA(n_components=2)
    # pca.fit(test_arr2)
    # print("Covariance matrix")
    # print(pca.get_covariance())

