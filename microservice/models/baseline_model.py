from constants import BASE_DIR, DATA_PART_TRAINING, TARGET_FEATURE_NAME

import os
import pandas as pd
import numpy as np



class BaselineModel:

    def __init__(self):
        os.chdir(BASE_DIR + f"{os.sep}proper_data{os.sep}data_version"
                            f"{DATA_PART_TRAINING}")
        self.users_data = pd.read_json(f"users{DATA_PART_TRAINING}.json")
        os.chdir("..")
        self.new_processed_data = pd.read_csv("processed_data.csv")


    def predict(self, single_item: dict):

        user_dfs = []

        # best accuracy
        look_back = 2

        for user_id in self.users_data.user_id:
            user_data = self.new_processed_data[
                    self.new_processed_data.user_id == user_id
                ].sort_values(['year', 'month'])

            for i in range(0, user_data.shape[0] - look_back):

                accumulated_value = 0
                for j in range(look_back):
                    accumulated_value += user_data.loc[user_data.index[i + j], TARGET_FEATURE_NAME]

                user_data.loc[user_data.index[i + look_back], f'SMA_{look_back}_before'] = np.round(
                    accumulated_value / look_back, 1)

            user_data.fillna(0, inplace=True)
            user_dfs.append(user_data[['year', 'month', 'user_id', 'money_monthly', f'SMA_{look_back}_before']])

        self.moving_average_data = pd.concat(user_dfs)

        mask_twenty = (self.moving_average_data['year'] == 2020) & (
                    (self.moving_average_data['month'] == 12) | (self.moving_average_data['month'] == 11))
        mask_twenty_one = (self.moving_average_data['year'] == 2021)

        self.moving_average_data = self.moving_average_data[mask_twenty | mask_twenty_one]

        prediction = self.moving_average_data[
            (self.moving_average_data['user_id'] == single_item['user_id']) & \
            (self.moving_average_data['year'] == single_item['year']) & \
            (self.moving_average_data['month'] == single_item['month'])
        ][f'SMA_{look_back}_before'].to_list()[0]

        return prediction
