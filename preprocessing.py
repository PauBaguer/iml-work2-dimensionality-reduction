import numpy as np
from sklearn import preprocessing as pre
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self, adult_df, vowel_df, pen_based_df, plot=False):

        self.pp_adult_df, self.pp_gs_adult_df, _ = self.preprocess_df(adult_df)
        self.pp_vowel_df, self.pp_gs_vowel_df, _ = self.preprocess_df(vowel_df)
        self.pp_pen_based_df, self.pp_gs_pen_based_df, _ = self.preprocess_df(pen_based_df)

        if plot:
            self.plot_data(self.pp_adult_df, self.pp_gs_adult_df, "Adult")
            self.plot_data(self.pp_vowel_df, self.pp_gs_vowel_df, "Vowel")
            self.plot_data(self.pp_pen_based_df, self.pp_gs_pen_based_df, "Pen-based")



    #####################################
    #             Missing values        #
    #####################################

    def erase_rows_with_missing_values(self,df):
        df_dropped_numerical = df.dropna(how='any')
        df_dropped_categorical = df_dropped_numerical
        for column in df_dropped_numerical:
            df_dropped_categorical = df_dropped_categorical.drop(df_dropped_categorical[df_dropped_categorical[column] == b'?'].index)

        return df_dropped_categorical


    def plot_data(self, X, labels, dataset_name):
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        print("Estimated number of points per cluster")
        points_per_cluster = {}
        unique_labels = set(labels)

        for l in unique_labels:
            points_per_cluster[l] = list(labels).count(l)
            print(f"Cluster {l}: {points_per_cluster[l]} points")

        plt.scatter(X[:,0:1], X[:,1:2], c=labels, s=3)


        # colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        # i=0
        # for row in X:
        #     plt.plot(
        #         row[0],
        #         row[1],
        #         ".",
        #         color=colors[labels[i]],
        #         markersize=3,
        #         zorder=labels[i]
        #     )
        #     # if i > 100:
        #     #     plt.show()
        #     #     return
        #     i = i+1





        plt.title(f"Preprocessed {dataset_name} dataset. Nº clusters: {n_clusters_}")
        plt.savefig(f"figures/preprocessing/{dataset_name}.png")
        plt.show()



    #####################################
    #   Main preprocessing functions    #
    #####################################
    def preprocess_df(self, df):
        prepped_df = df#erase_rows_with_missing_values(df)

        classification_goldstandard_cols = ["class", "a17", "Class"] # The columns to take out of preprocessing bc they are the final gold standard classification.
        goldstandard_col = []
        categorical_cols = []
        numeric_cols = []
        for col in prepped_df:
            if col in classification_goldstandard_cols:
                goldstandard_col.append(col)
                break
            col_type = type(df[col].values[0])
            if col_type == bytes:
                # Categorical data
                categorical_cols.append(col)

            elif col_type == np.float64:
                # Numerical data
                numeric_cols.append(col)
            else:
                print('Check other type!!')

        # transform bytes to string.
        str_df = prepped_df.select_dtypes([np.object])
        str_df = str_df.stack().str.decode("utf-8").unstack()
        for col in str_df:
            prepped_df[col] = str_df[col]

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # fill missing values with most frequent
            ("encoder", pre.OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False)),
            # ("min-max-scaler", MinMaxScaler())
        ])

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")), # fill missing values with the median
            ("scaler", StandardScaler()),
            #  ("min-max-scaler", MinMaxScaler())
        ])

        preprocessor = ColumnTransformer(sparse_threshold=0, transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])

        preprocessor.fit(prepped_df)
        transformed_df = preprocessor.transform(prepped_df)

        print()

        goldstandard_preprocessor = pre.LabelEncoder()

        goldstandard_preprocessor.fit(prepped_df[goldstandard_col].values.ravel())
        transformed_goldstandard_col_df = goldstandard_preprocessor.transform(prepped_df[goldstandard_col].values.ravel())


        return transformed_df, transformed_goldstandard_col_df, preprocessor