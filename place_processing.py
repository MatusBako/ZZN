from datetime import datetime, timedelta

import pandas as pd
from pandas.core.frame import DataFrame

from typing import List, Tuple


seconds_in_day = 60*60*24


def extract_time(interval: str) -> Tuple[datetime, datetime]:
    """ Parse time interval to datetime objects.

    Args:
        interval (str):  time interval of work shift

    Returns:
         tuple(datetime, datetime): start and end of shift
    """
    times = interval.split('-')

    # some intervals only contain one time ... but why
    if len(times) != 2:
        raise AttributeError

    start, stop = times

    # convert string to Datetime object
    start = datetime.strptime(start, "%H:%M")
    stop = datetime.strptime(stop, "%H:%M")

    if start >= stop:
        # opened till 0:00 the next day
        if stop == datetime.strptime("00:00", "%H:%M"):
            stop += timedelta(days=1)
        # times are switched ... but why
        else:
            stop = datetime.strptime("00:00", "%H:%M") + timedelta(days=1)
    return start, stop


def get_place_hours() -> (DataFrame, List[int]):
    """ Parse and return Places with their respective opening hours

    Returns:
        DataFrame: placeID with it's opening hours
        List: list of place IDs which were removed from DataFrame (error in csv)
    """
    df = pd.read_csv("data/chefmozhours4.csv", encoding='utf-8')
    df.placeID = df.placeID.astype(int)
    df.drop_duplicates(subset=df.columns, keep='first', inplace=True)

    # create new dataframe
    columns = ['placeID', 'hours_week', 'hours_sat', 'hours_sun']
    new_df = DataFrame(columns=['placeID', 'hours_week', 'hours_sat', 'hours_sun'])
    new_df['placeID'] = new_df['placeID'].astype(int)
    last_place_id = None

    for row in df.itertuples():
        place_id = row[1]
        days = row[-1]
        total_time = timedelta()

        # extract time intervals
        intervals = row[2].split(';')
        intervals = list(filter(lambda x: len(x) > 0, intervals))

        # new placeID encountered
        if place_id != last_place_id:
            # append empty row to dataframe (placeID and empty hours)
            new_row = [place_id] + [0.] * (len(columns) - 1)
            new_df = new_df.append(pd.DataFrame([new_row], columns=columns), ignore_index=True)
            last_place_id = place_id

        # iterate over all opening times in one day
        for interval in intervals:
            try:
                start, stop = extract_time(interval)
            except AttributeError:
                continue

            # sum all time intervals in day
            total_time += stop - start

        if 'Mon' in days:
            column = 'hours_week'

            # time is specified for all days once possibly using more intervals
            if len(intervals) < 5:
                total_time *= 5

            total_time = total_time.total_seconds() / (seconds_in_day * 5)
        elif 'Sat' in days:
            column = 'hours_sat'
            total_time = total_time.total_seconds() / seconds_in_day
        elif 'Sun' in days:
            column = 'hours_sun'
            total_time = total_time.total_seconds() / seconds_in_day
        else:
            raise Exception("Row in table with wrong day!")

        new_df.at[new_df.index.max(), column] = total_time

    error_rows = new_df.loc[
        (new_df['hours_week'] > 1.1)
        | (new_df['hours_sat'] > 1.1)
        | (new_df['hours_sun'] > 1.1)
        ]
    error_place_ids = error_rows.placeID.unique()

    # remove lines with errors
    new_df = new_df[~new_df.placeID.isin(error_place_ids)]

    return new_df


def get_place_secondary_df(name: str, value_column: str) -> DataFrame:
    """ Parse csv file with secondary info about place (payment, cuisine)

    Args:
        name (str): table name without prefix (chefmoz) and suffix (.csv)
        value_column (str): name of column that contains wanted values

    Returns:
        DataFrame:
    """
    df = pd.read_csv("data/chefmoz" + name + ".csv", encoding='utf-8')

    # create list of columns
    columns = ['placeID'] + list("R" + name + "_" + df[value_column].unique())
    new_df = pd.DataFrame(columns=columns)

    for column in new_df.columns[1:]:
        new_df[column].astype(int)

    # iterate over every row
    for user in df['placeID'].unique():
        # create row with user ID and zero values in other columns
        row = [user] + [0] * (len(columns) - 1)

        # append row to dataframe
        new_df = new_df.append(pd.DataFrame([row], columns=columns), ignore_index=True)

        # set every cuisine user has (likes)
        for value in df.loc[df.placeID == user][value_column]:
            new_df.at[new_df.index.max(), "R" + name + "_" + value] = 1

    new_df.placeID = new_df.placeID.astype('int64')
    return new_df


def get_places() -> DataFrame:
    """ Get data describing places.

    Returns:
         Dataframe: DF containing every place with its respective info.
    """
    df = pd.read_csv('./data/geoplaces2.csv', encoding='utf-8')

    # drop useless columns
    df.drop(columns=['the_geom_meter', 'name', 'address',
                     'city', 'state', 'country', 'fax',
                     'zip', 'url', 'accessibility', 'franchise',
                     'other_services'],
            inplace=True)

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

    categorical_columns = [column for column in df.columns if df[column].dtype.name == 'object']

    for column in categorical_columns:
        df[column] = df[column].astype('category')

    df_cuisine = get_place_secondary_df('cuisine', 'Rcuisine')
    df_payment = get_place_secondary_df('accepts', 'Rpayment')
    df_hours = get_place_hours()

    payment_columns = list(filter(lambda x: x.startswith("Raccepts_"), df_payment.columns))

    # some restaurants don't have specified payment ... but why
    # left join payment options and set cash option
    new_df = df.merge(df_payment, on='placeID', how='left')
    new_df[payment_columns] = new_df[payment_columns].fillna(0)
    new_df['Raccepts_cash'] = 1

    # left join cuisines and fill missing values with 0
    new_df = new_df.merge(df_cuisine, on='placeID', how='left')
    cuisine_columns = list(filter(lambda x: "Rcuisine" in x, new_df.columns))
    new_df[cuisine_columns] = new_df[cuisine_columns].fillna(0)

    new_df = new_df.merge(df_hours, on='placeID', how='inner')

    return new_df
