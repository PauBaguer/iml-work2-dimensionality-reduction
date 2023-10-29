from scipy.io import arff
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import Preprocessing
import birch, kmeans

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

