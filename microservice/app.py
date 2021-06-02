import uvicorn
import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from BasicModel import BasicModel

class ModelSpecies(BaseModel):
	user_id: int
	year: int
	month: int
	total_web_events: int
	buying_events: int
	num_dropped_monthly: float
	sum_after_discount: float
	
	
class Model:
	def __init__(self):
		self.model_fname_ = 'model.pkl'
		try:
			self.model = joblib.load(self.model_fname_)
		except Exception as _:
			# tutaj wrzucic kod uczenia modelu
			self.model = self._train_our_model()
			joblib.dump(self.model, self.model_fname_)
            
	def _train_our_model(self):
		# tutaj powinno się wrzucić kod trenujący ostateczny model i zwracający go
		processed_data = pd.read_csv('processed_data.csv')
		data_to_learning = processed_data.dropna(subset=['next_months_sum'])
		msk = np.random.rand(len(data_to_learning)) < 0.8
		train = data_to_learning[msk]
		inputs_train = train.dropna(subset=['next_months_sum'])
		columns = inputs_train.columns
		columns = columns.drop('next_months_sum')
		columns = columns.drop('city')
		columns = columns.drop('Unnamed: 0')
		print(columns)
		x = inputs_train[columns].values
		y = train.next_months_sum.values
        
		forest = RandomForestRegressor(n_estimators=500,
							criterion='mse',
							random_state=1,
							n_jobs=-1,
							min_samples_leaf=2)
		forest.fit(x,y)
		return forest
		
		
	def predict_species(self, user_id, year, month, total_web_events, buying_events, num_dropped_monthly, sum_after_discount):
		data_in = [[user_id, year, month, total_web_events, buying_events, num_dropped_monthly, sum_after_discount]]
		prediction = self.model.predict(data_in)
		# probability = self.model.predict_proba(data_in).max()
		return prediction[0]#, probability


processed_data = pd.read_csv('processed_data.csv')
times = processed_data['year'] * 100 + processed_data['month']
max_year = int(times.max()/100)
max_month = int(times.max()%100)

app = FastAPI()
model = Model()
basicModel = BasicModel()


@app.get('/predict')
def predict_species(user_id: int, model_type=None):
	data = processed_data[(processed_data.year == max_year) & (processed_data.month == max_month) & (processed_data.user_id == user_id)]
	save_to_file = False
	if model_type == None:
		save_to_file = True
		if user_id % 2 == 0:
			model_type = "BASE"
		else:
			model_type = "OUR"
	if model_type == "OUR":
		prediction = model.predict_species(user_id, max_year, max_month, data['total_web_events'], int(data['buying_events']), (data['num_dropped_monthly']), float(data['sum_after_discount']))
	else:
		prediction = basicModel.predict(data)	
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
    
 
