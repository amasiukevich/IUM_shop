import os
import pandas as pd
import numpy as np

import itertools

from constants import DATA_PART_TRAINING, BASE_DIR

def get_valid_user_ids():

    os.chdir(BASE_DIR)
    users_data = pd.read_json(
        f'proper_data{os.sep}data_version2{os.sep}users{DATA_PART_TRAINING}.json', lines=True)
    return list(users_data['user_id'].unique())


def get_valid_years():

    os.chdir(BASE_DIR)
    events_data = pd.read_json(
        f'proper_data{os.sep}data_version2{os.sep}sessions{DATA_PART_TRAINING}.json',
        lines=True
    )
    events_data['year'] = events_data.timestamp.year

    return list(events_data['year'].unique())


def get_max_year_month():

    os.chdir(BASE_DIR)
    events_data = pd.read_json(f'proper_data{os.sep}data_version2{os.sep}sessions{DATA_PART_TRAINING}.json')
    max_timestamp = events_data.timestamp.max()

    max_year, max_month = max_timestamp.year, max_timestamp.month

    return max_year, max_month


def get_features(user_id, month, year):

    os.chdir(BASE_DIR)
    data = pd.read_csv(f'proper_data{os.sep}processed_data.csv')
    user_data = data[(data['user_id'] == user_id) & \
                     (data['year'] == year) & \
                     (data['month'] == month)]


    features_exist = any([dict(user_data[feature]) for feature in user_data])
    if features_exist:
        results = user_data[[
            "buying_sessions_prev_month",
            "buying_sessions_MA3",
            "buying_ratio_prev_month",
            "money_monthly_MA3",
            "buying_ratio_MA3",
            "prev_month_spendings"
        ]]



        return results, features_exist
    else:
        return None, features_exist



def extract_features(data_folder):

    users_data = pd.read_json(data_folder + f"{os.sep}users{DATA_PART_TRAINING}.json")
    products_data = pd.read_json(data_folder + f"{os.sep}products{DATA_PART_TRAINING}.json")
    events_data = pd.read_json(data_folder + f"{os.sep}sessions{DATA_PART_TRAINING}.json")
    deliveries_data = pd.read_json(data_folder + f"{os.sep}deliveries{DATA_PART_TRAINING}.json")


    max_timestamp = events_data.timestamp.max()

    max_year_month = max_timestamp.year, max_timestamp.month

    processed_data = create_basic_df(events_data)

    buying_features_data = get_buying_features(events_data)
    processed_data = pd.merge(processed_data, buying_features_data, how='left',
                              on=['year', 'month', 'user_id'])
    processed_data['buying_sessions'].fillna(0, inplace=True)
    processed_data['buying_ratio'].fillna(0, inplace=True)


    monthly_deals = construct_target_var(products_data, events_data)
    processed_data = pd.merge(processed_data, monthly_deals, how='left',
                              on=['year', 'month', 'user_id'])

    processed_data['money_monthly'].fillna(0, inplace=True)


    new_processed_data = construct_user_features(users_data, processed_data)


    lifetime_values = split_rich_users(new_processed_data)
    new_processed_data = pd.merge(new_processed_data, lifetime_values[['user_id', 'is_rich']], on='user_id',
                                  how='left')

    mask_twenty = (new_processed_data['year'] == 2020) & (
                (new_processed_data['month'] == 12) | (new_processed_data['month'] == 11))
    mask_twenty_one = (new_processed_data['year'] == 2021)


    new_processed_data = new_processed_data[mask_twenty | mask_twenty_one]


    new_processed_data.to_csv('proper_data/processed_data.csv', index=False)


    return new_processed_data


def create_basic_df(events_data):

    events_data['year'] = events_data.timestamp.dt.year
    events_data['month'] = events_data.timestamp.dt.month

    years = list(events_data.year.unique())
    months = list(events_data.month.unique())
    user_ids = list(events_data.user_id.unique())

    triplets = []
    for triplet in itertools.product(years, months, user_ids):
        triplets.append(triplet)

    processed_data = pd.DataFrame(
        triplets,
        columns=['year', 'month', 'user_id']
    )

    processed_data.drop(
        processed_data[(processed_data['year'] == 2021) & (processed_data['month'] > 4)].index,
        inplace=True
    )

    return processed_data


def get_buying_features(events_data):

    all_events = events_data.groupby(['user_id', 'year', 'month']).aggregate({"session_id": "count"}) \
        .rename(columns={"session_id": "all_sessions"}) \
        .reset_index()

    buying_events = events_data[events_data['event_type'] == 'BUY_PRODUCT'] \
        .groupby(['user_id', 'year', 'month']) \
        .aggregate({"session_id": "count"}) \
        .rename(columns={"session_id": "buying_sessions"}) \
        .reset_index()

    events_ratio = pd.merge(all_events, buying_events, how="left", on=["year", "month", "user_id"])

    events_ratio['buying_sessions'].fillna(0, inplace=True)
    events_ratio['buying_ratio'] = round(events_ratio['buying_sessions'] / events_ratio['all_sessions'], 4)

    return events_ratio[['year', 'month', 'user_id', 'buying_sessions', 'buying_ratio']]


def construct_target_var(products_data, events_data):

    buying_sessions = events_data[events_data['event_type'] == "BUY_PRODUCT"]
    deals = pd.merge(buying_sessions, products_data, how="left", on=['product_id'])
    deals['final_price'] = deals['price'] * (1 - deals['offered_discount'] * 0.01)

    monthly_deals = deals.groupby(['year', 'month', 'user_id']) \
        .aggregate({"final_price": "sum"}) \
        .rename(columns={"final_price": "money_monthly"}) \
        .reset_index()

    return monthly_deals



def construct_user_features(users_data, processed_data):

    user_dfs = []

    look_back = 3
    look_back_money = 3
    for user_id in users_data.user_id:

        user_data = processed_data[processed_data.user_id == user_id].sort_values(['year', 'month'])

        for i in range(0, user_data.shape[0] - look_back):
            accumulated_value = 0
            for j in range(look_back):
                accumulated_value += user_data.loc[user_data.index[i + j], 'buying_sessions']

            user_data.loc[user_data.index[i + look_back], f'buying_sessions_MA{look_back}'] = np.round(
                accumulated_value / look_back, 1)

        for i in range(0, user_data.shape[0] - look_back):
            accumulated_value = 0
            for j in range(look_back):
                accumulated_value += user_data.loc[user_data.index[i + j], 'buying_ratio']

            user_data.loc[user_data.index[i + look_back], f'buying_ratio_MA{look_back}'] = np.round(
                accumulated_value / look_back, 3)

        for i in range(0, user_data.shape[0] - look_back_money):
            accumulated_value = 0
            for j in range(look_back_money):
                accumulated_value += user_data.loc[user_data.index[i + j], 'money_monthly']

            user_data.loc[user_data.index[i + look_back_money], f'money_monthly_MA{look_back_money}'] = np.round(
                accumulated_value / look_back_money, 1)

        user_data[f'buying_sessions_MA{look_back}'].fillna(0, inplace=True)
        user_data[f'buying_ratio_MA{look_back}'].fillna(0, inplace=True)
        user_data[f'money_monthly_MA{look_back_money}'].fillna(0, inplace=True)

        user_data['prev_month_spendings'] = user_data['money_monthly'].shift(1)
        user_data['prev_month_spendings'].fillna(0, inplace=True)

        user_data['prev_second_month_spendings'] = user_data['money_monthly'].shift(2)
        user_data['prev_second_month_spendings'].fillna(0, inplace=True)

        user_data['prev_third_month_spendings'] = user_data['money_monthly'].shift(3)
        user_data['prev_third_month_spendings'].fillna(0, inplace=True)

        user_data['buying_sessions_prev_month'] = user_data['buying_sessions'].shift(1)
        user_data['buying_sessions_prev_month'].fillna(0, inplace=True)

        user_data['buying_sessions_p_second_month'] = user_data['buying_sessions'].shift(2)
        user_data['buying_sessions_p_second_month'].fillna(0, inplace=True)

        user_data['buying_sessions_p_third_month'] = user_data['buying_sessions'].shift(3)
        user_data['buying_sessions_p_third_month'].fillna(0, inplace=True)

        user_data['buying_ratio_prev_month'] = user_data['buying_ratio'].shift(1)
        user_data['buying_ratio_prev_month'].fillna(0, inplace=True)

        user_data['buying_ratio_p_second_month'] = user_data['buying_ratio'].shift(2)
        user_data['buying_ratio_p_second_month'].fillna(0, inplace=True)

        user_data['buying_ratio_p_third_month'] = user_data['buying_ratio'].shift(3)
        user_data['buying_ratio_p_third_month'].fillna(0, inplace=True)

        user_dfs.append(user_data)

    new_processed_data = pd.concat(user_dfs)

    return new_processed_data


def split_rich_users(new_processed_data):

    user_lifetime_value = new_processed_data \
        .groupby('user_id') \
        .aggregate({"money_monthly": "sum"}) \
        .rename(columns={"money_monthly": "lifetime_value"})

    riches = []

    quantile_eighty_five = new_processed_data \
        .groupby('user_id') \
        .aggregate({"money_monthly": 'sum'}) \
        .quantile(0.85) \
        .to_list()[0]

    for idx, row_data in user_lifetime_value.iterrows():

        row_data = dict(row_data)

        lf_value = row_data['lifetime_value']

        if lf_value > quantile_eighty_five:
            riches.append(1)
        else:
            riches.append(0)


    user_lifetime_value['is_rich'] = riches
    user_lifetime_value = user_lifetime_value.reset_index()

    return user_lifetime_value