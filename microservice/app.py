import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

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
			self.model = self._train_model()
			joblib.dump(self.model, self.model_fname_)
            
	def _train_model(self):
		# tutaj powinno się wrzucić kod trenujący ostateczny model i zwracający go
		processed_data = pd.read_csv('../processed_data.csv')
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


app = FastAPI()
model = Model()

@app.get('/predict')
def predict_species(modelSpecies: ModelSpecies):
    data = modelSpecies.dict()
    prediction = model.predict_species(
        data['user_id'], data['year'], data['month'], data['total_web_events'], data['buying_events'], data['num_dropped_monthly'], data['sum_after_discount']
    )
    return {
        'prediction': prediction,
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
 
