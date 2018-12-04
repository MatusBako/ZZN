#!/usr/bin/env python3

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from user_processing import get_users
from place_processing import get_places

import h2o
import pandas as pd

# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori

from scipy.stats import zscore

from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 200)

def heatmap():
    df = get_users()
    df_ratings = pd.read_csv("data/rating_final.csv")

    # get place with most ratings
    place_id = df_ratings.placeID.value_counts().idxmax()

    # filter ratings of given place
    df_ratings = df_ratings[df_ratings.placeID == place_id]

    # join user profile with rating on user ID
    df = df.merge(df_ratings[['userID', 'rating']], how='inner', on='userID')

    columns = ['birth_year', 'weight', 'height', 'rating']
    data = np.array(df[columns].corr()).round(2)

    ax = plt.gca()
    im = ax.imshow(data, cmap='RdBu')  # Create colorbar
    im.set_clim(-1, 1)
    cbar = ax.figure.colorbar(im, ax=ax, )

    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # give text to each point in map
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, data[i, j], ha="center", va="center", color=("w" if i == j else "k"))

    plt.tight_layout()
    #plt.show()
    plt.savefig("heatmap.png")
    # TODO: save plot


if __name__ == "__main__":
    #train_h2o()
    heatmap()
    pass
