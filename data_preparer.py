import json
import itertools
import pandas as pd
import numpy as np

def prepare_data():
	users_data = pd.read_json("proper_data/users2.json")
	deliveries_data = pd.read_json("proper_data/deliveries2.json")
	events_data = pd.read_json("proper_data/sessions2.json")
	products_data = pd.read_json("proper_data/products2.json")

	max_timestamp = events_data.timestamp.max()

	max_year_month = max_timestamp.year, max_timestamp.month

	events_data['year'] = events_data.timestamp.dt.year
	events_data['month'] = events_data.timestamp.dt.month



	years = list(events_data.year.unique())
	months = list(events_data.month.unique())
	user_ids = list(events_data.user_id.unique())

	triplets = []
	for triplet in itertools.product(years, months, user_ids):
		triplets.append(triplet)
		
	processed_data = pd.DataFrame(triplets, columns=['year', 'month', 'user_id'])
	processed_data.drop(processed_data[(processed_data['year'] == 2021) & (processed_data['month'] > 4)].index, inplace=True)

	all_events = events_data.groupby(['user_id', 'year', 'month']).aggregate({"session_id": "count"}) \
		.rename(columns={"session_id": "all_sessions"}) \
		.reset_index()
		
	buying_events = events_data[events_data['event_type'] == 'BUY_PRODUCT'] \
		.groupby(['user_id', 'year', 'month']) \
		.aggregate({"session_id": "count"}) \
		.rename(columns={"session_id": "buying_sessions"}) \
		.reset_index()
		
	events_ratio = pd.merge(all_events, buying_events, how="left", on=["year", "month", "user_id"])

	# all_events
	# buying_events

	events_ratio['buying_sessions'].fillna(0, inplace=True)

	events_ratio['buying_ratio'] = round(events_ratio['buying_sessions'] / events_ratio['all_sessions'], 4)


	# merging with processed_data
	processed_data = pd.merge(processed_data, events_ratio.drop("all_sessions", axis=1), how='left', on=['year', 'month', 'user_id'])
	processed_data['buying_sessions'].fillna(0, inplace=True)
	processed_data['buying_ratio'].fillna(0, inplace=True)
	
	buying_sessions = events_data[events_data['event_type'] == "BUY_PRODUCT"]
	deals = pd.merge(buying_sessions, products_data, how="left", on=['product_id'])
	deals['final_price'] = deals['price'] * (1 - deals['offered_discount'] * 0.01)

	monthly_deals = deals.groupby(['year', 'month', 'user_id']) \
		.aggregate({"final_price": "sum"}) \
		.rename(columns={"final_price": "money_monthly"}) \
		.reset_index()

	processed_data = pd.merge(processed_data, monthly_deals, how='left', on=['year', 'month', 'user_id'])
	processed_data['money_monthly'].fillna(0, inplace=True)
	
	user_dfs = []

	look_back = 3
	look_back_money = 3
	for user_id in users_data.user_id:
		
		user_data = processed_data[processed_data.user_id == user_id].sort_values(['year', 'month'])
		
		
		for i in range(0, user_data.shape[0] - look_back):
		    accumulated_value = 0
		    for j in range(look_back):
		        accumulated_value += user_data.loc[user_data.index[i + j], 'buying_sessions']
		    
		    user_data.loc[user_data.index[i + look_back], f'buying_sessions_MA{look_back}'] = np.round(accumulated_value / look_back, 1)
		
		
		for i in range(0, user_data.shape[0] - look_back):
		    accumulated_value = 0
		    for j in range(look_back):
		        accumulated_value += user_data.loc[user_data.index[i + j], 'buying_ratio']
		    
		    user_data.loc[user_data.index[i + look_back], f'buying_ratio_MA{look_back}'] = np.round(accumulated_value / look_back, 3)
		    
		
		for i in range(0, user_data.shape[0] - look_back_money):
		    accumulated_value = 0
		    for j in range(look_back_money):
		        accumulated_value += user_data.loc[user_data.index[i + j], 'money_monthly']
		    
		    user_data.loc[user_data.index[i + look_back_money], f'money_monthly_MA{look_back_money}'] = np.round(accumulated_value / look_back_money, 1)
		    
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
	
	quantile_eighty_five = new_processed_data.groupby('user_id').aggregate({"money_monthly": 'sum'}).quantile(0.85).to_list()[0]
	
	uantile_eighty_five = new_processed_data.groupby('user_id').aggregate({"money_monthly": 'sum'}).quantile(0.85).to_list()[0]
	
	user_lifetime_value = new_processed_data.groupby('user_id').aggregate({"money_monthly": "sum"}).rename(columns={"money_monthly": "lifetime_value"})

	riches = []

	for idx, row_data in user_lifetime_value.iterrows():
		
		row_data = dict(row_data)
		
		lf_value = row_data['lifetime_value']
		
		if lf_value > quantile_eighty_five:
		    riches.append(1)
		else:
		    riches.append(0)
		    


	# user_lifetime_value['is_rich'] = [not elem for elem in dummy]
	user_lifetime_value['is_rich'] = riches
	user_lifetime_value = user_lifetime_value.reset_index()

	new_processed_data = pd.merge(new_processed_data, user_lifetime_value[['user_id', 'is_rich']], on='user_id', how='left')
	
	
	new_processed_data.to_csv('processed_data.csv')
		
		
prepare_data()
