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
