import pandas as pd
from pandas.core.frame import DataFrame

from typing import Tuple


def get_user_secondary_df(name: str, value_column: str) -> DataFrame:
    """

    Args:
        name (str):
        value_column (str):

    Returns:
        DataFrame:
    """
    df = pd.read_csv("data/user" + name + ".csv", encoding='utf-8')

    # create list of columns
    columns = ['userID'] + list("U" + name + "_" + df[value_column].unique())
    new_df = pd.DataFrame(columns=columns)

    # TODO: change type of columns to int
    for column in new_df.columns[1:]:
        new_df[column].astype(int)

    # iterate over every row
    for user in df['userID'].unique():
        # create row with user ID and zero values in other columns
        row = [user] + [0] * (len(columns) - 1)

        # append row to dataframe
        new_df = new_df.append(pd.DataFrame([row], columns=columns), ignore_index=True)

        # set every cuisine user has (likes)
        for value in df.loc[df.userID == user][value_column]:
            new_df.at[new_df.index.max(), "U" + name + "_" + value] = 1

    return new_df


def get_users() -> DataFrame:
    """ Get users dataframe with respective data.

    Returns:
        Dataframe: Expected data.
    """
    df = pd.read_csv("data/userprofile.csv", encoding='utf-8')

    # drop useless columns
    df.drop(columns=['color', 'interest', 'personality', 'height',
                     'activity', 'religion', 'ambience'],
            inplace=True)

    user_ids = set()

    # remove '?' values
    for column in df.columns:
        if df[column].dtype.name == 'object':
            for user_id in df.loc[df[column] == "?"].userID:
                user_ids.add(user_id)
            df = df[df[column] != '?']

    df.smoker = df.smoker.replace({'true': 1, 'false': 0})

    # select categorical column names
    categorical_columns = [column for column in df.columns
                           if df[column].dtype.name == 'object'
                           if column not in ['userID', 'smoker']]

    # replace categorical columns with one hot encoding
    for column_name in categorical_columns:
        dummies = pd.get_dummies(df[column_name])

        for dummy_column_name in dummies.columns:
            df[column_name + "_" + dummy_column_name] = dummies[dummy_column_name]

        df.drop(columns=[column_name], inplace=True)

    df_cuisine = get_user_secondary_df('cuisine', 'Rcuisine')
    df_payment = get_user_secondary_df('payment', 'Upayment')

    # merge dataframes on user ID
    new_df = df.merge(df_payment, on='userID', how='inner') \
            .merge(df_cuisine, on='userID', how='inner')

    return new_df
