#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from user_processing import get_users
from place_processing import get_places
from transforms import transform_coords, transform_payments, transform_cuisines


def example_all_ratings():
    seed = int(time())  #42
    np.random.seed(seed)

    # load dataframes
    user_df = get_users()
    places_df = get_places()
    ratings_df = pd.read_csv("data/rating_final.csv", encoding='utf-8')\
                   .drop(columns=['food_rating', 'service_rating'])

    # merge all tables together
    df = ratings_df.merge(user_df, how='inner', on='userID') \
                   .merge(places_df, how='inner', on='placeID')

    df = transform_cuisines(df)
    df = transform_payments(df)
    df = transform_coords(df)

    # drop row identifiers
    df.drop(columns=['userID', 'placeID'], inplace=True)

    x = df.drop(columns=['rating'])
    y = df.rating


    means = []

    for i in range(100):
        # split dataset
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.9)

        # train classifier
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(xtrain, ytrain)

        # evaluate classifier
        predictions = clf.predict(xtest)
        errors = np.array(abs(predictions - ytest))
        mean = np.mean(errors.flatten())
        means.append(mean)

    mean = np.mean(means)
    std = np.std(means)

    print("Mean:", mean, "\nStd:", std)
    print(means)

    #print('Mean Absolute Error: \"', round(np.mean(errors), 2), '\".')
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
    plt.show()

example_all_ratings()
