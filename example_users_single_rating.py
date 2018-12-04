#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from user_processing import get_users


def example_single_rating():
    user_df = get_users()

    # drop useless columns
    prefixes = ['latitude', 'longitude']
    useless_cols = [col for col in user_df.columns if col.split('_')[0] in prefixes]
    user_df.drop(columns=useless_cols, inplace=True)

    df_ratings = pd.read_csv("data/rating_final.csv", encoding='utf-8')

    # get place with most ratings
    place_id = df_ratings.placeID.value_counts().idxmax()

    # filter ratings and join user profile with rating on user ID
    df_ratings = df_ratings[df_ratings.placeID == place_id]
    df = user_df.merge(df_ratings[['userID', 'rating']], how='inner', on='userID')

    x = df.drop(columns=['rating', 'userID'])
    y = df.rating

    means = []

    for i in range(100):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.9)

        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(xtrain, ytrain)

        predictions = clf.predict(xtest)
        errors = abs(predictions - ytest)
        mean = np.mean(errors)
        means.append(mean)

    mean = np.mean(means)
    std = np.std(means)

    print("Mean:", mean, "\nStd:", std)
    print(means)

    #print('Mean Absolute Error: \"', round(mean, 2), '\".')
    # print(list((xtest, predictions, ytest))[:10])

    features = x.columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)

    # filter by importances > 0
    features = np.array([features[index] for index in indices if importances[index] > 0])[-10:]
    importances = np.array([importances[index] for index in indices if importances[index] > 0])[-10:]

    # show feature importance plot
    plt.barh(range(len(importances)), importances[:], color='b', align='center')
    plt.yticks(range(len(features)), features)
    #plt.title('Feature Importances')
    #plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig("importance.png")

example_single_rating()
