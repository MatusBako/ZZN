import pandas as pd

def coords_to_dist(lat1, lat2, lon1, lon2):
    from math import sin, cos, sqrt, atan2

    radius = 6373

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return radius * c


def transform_cuisines(df: pd.DataFrame) -> pd.DataFrame:
    cuisines = [column[len("Rcuisine_"):] for column in df.columns if column.startswith("Rcuisine_")]
    df['cuisines_common'] = 0

    for row in df.itertuples():
        cuisines_common = 0
        row_index = row[0]

        for cuisine in cuisines:
            if df["Rcuisine_" + cuisine][row_index] and df["Ucuisine_" + cuisine][row_index]:
                cuisines_common += 1

        df['cuisines_common'][row_index] = cuisines_common

    all_cuisines = [column for column in df.columns if "cuisine_" in column]

    return df.drop(columns=all_cuisines)


def transform_payments(df: pd.DataFrame) -> pd.DataFrame:
    payments = [column[len("Upayment_"):] for column in df.columns if column.startswith("Upayment_")]
    df['payments_common'] = 0

    for row in df.itertuples():
        payments_common = 0
        row_index = row[0]

        for payment in payments:
            if df["Raccepts_" + payment][row_index] and df["Upayment_" + payment][row_index]:
                payments_common += 1

        df['payments_common'][row_index] = payments_common

    all_payments = [column for column in df.columns if "Upayment_" in column or "Raccepts_" in column]

    return df.drop(columns=all_payments)


def transform_coords(df: pd.DataFrame) -> pd.DataFrame:
    df['distance'] = 0.

    for row in df.itertuples():
        index = row[0]
        dist = coords_to_dist(row.latitude_x, row.latitude_y, row.longitude_x, row.longitude_y)
        df.at[index, 'distance'] = dist

    return df.drop(columns=['latitude_x', 'latitude_y', 'longitude_x', 'longitude_y'])
