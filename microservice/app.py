import uvicorn
import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
	
class BasicModel:
	def __init__(self):
		self.model_fname_ = 'BasicModel.pkl'
		try:
			self.model = joblib.load(self.model_fname_)
		except Exception as _:
			# tutaj wrzucic kod uczenia modelu
			self.model = self._train_our_model()
			joblib.dump(self.model, self.model_fname_)
            
	def _train_our_model(self):
		linear_model = LinearRegression()
		linear_model.fit(X, y)

		scores_linear = cross_val_score(
			linear_model,
			X=X,
			y=y,
			scoring='neg_mean_absolute_error',
			cv=5
		)
		return linear_model
		
		
	def predict_species(self, data_features):
		prediction = self.model.predict(data_features)
		return prediction[0]
	
	
class Model:
	def __init__(self):
		self.model_fname_ = 'Model.pkl'
		try:
			self.model = joblib.load(self.model_fname_)
		except Exception as _:
			# tutaj wrzucic kod uczenia modelu
			self.model = self._train_our_model()
			joblib.dump(self.model, self.model_fname_)
            
	def _train_our_model(self):
		svr_model = SVR(kernel='rbf', C=10000)
		svr_model.fit(X, y)

		scores_svr = cross_val_score(
			svr_model,
			X=X,
			y=y,
			scoring='neg_mean_absolute_error',
			cv=5
		)
		return svr_model
		
		
	def predict_species(self, data_features):
		prediction = self.model.predict(data_features)
		return prediction[0]

look_back = 3
look_back_money = 3
processed_data = pd.read_csv('processed_data.csv')
times = processed_data['year'] * 100 + processed_data['month']
max_year = int(times.max()/100)
max_month = int(times.max()%100)

mask_twenty = (processed_data['year'] == 2020) & ((processed_data['month'] == 12) | (processed_data['month'] == 11))
mask_twenty_one = (processed_data['year'] == 2021)

data = processed_data[mask_twenty | mask_twenty_one]

data.drop(['year', 'month', 'user_id', 'buying_sessions', 'buying_ratio'], axis=1, inplace=True)

useful_features = ['is_rich', f'buying_sessions_MA{look_back}', f'money_monthly_MA{look_back_money}', 'buying_sessions_prev_month', 'buying_ratio_prev_month','prev_month_spendings']

data_to_train = data[useful_features + ['money_monthly']]
X = data_to_train[useful_features]

y = data_to_train['money_monthly']

app = FastAPI()
model = Model()
basicModel = BasicModel()


@app.get('/predict')
def predict_species(user_id: int, model_type=None):
	data_features = processed_data[(processed_data.year == max_year) & (processed_data.month == max_month) & (processed_data.user_id == user_id)]
	data_features = data_features[useful_features]
	save_to_file = False
	if model_type == None:
		save_to_file = True
		if user_id % 2 == 0:
			model_type = "BASE"
		else:
			model_type = "OUR"
	if model_type == "OUR":
		prediction = model.predict_species(data_features)
	else:
		prediction = basicModel.predict_species(data_features)	
	result = {
		'prediction_time': datetime.now().strftime("%m/%d/%Y, %H:%M:%S:%f"),
		'user_id': user_id,
		'year': max_year,
		'month': max_month,
		'model_type': model_type,
        'prediction': prediction
    }
	if save_to_file:
		file_object = open('logs/predictions.json', 'a')
		json.dump(result, file_object)
		file_object.write("\n")
		file_object.close()
	return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
 
