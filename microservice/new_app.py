from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models.baseline_model import BaselineModel
from models.trained_model import TrainedModel

from training.feature_extractor import get_features, get_max_year_month
from datetime import datetime

import json
import uvicorn


import pandas as pd
import numpy as np

# from training.feature_extractor import extract_features

app = FastAPI()
baseline_model = BaselineModel()
trained_model = TrainedModel()


class RequestData(BaseModel):
    user_id: int
    year: Optional[int] = None
    month: Optional[int] = None
    model_type: str



def validate_user_id(user_id: int):
    pass


def validate_month_year(month: Optional[int] = None, year: Optional[int] = None):
    pass


def validate_model_type(model_type: str = None):

    if model_type not in ["BASE", "TRAINED"]:
        pass
        # error message here


# endpoints
@app.get("/hi")
def hi():
    return "Hi there"


@app.get('/predict')
def predict(request_data: RequestData):

    request_data_dict = request_data.dict()

    user_id = request_data_dict['user_id']
    year = request_data_dict['year']
    month = request_data_dict['month']
    model_type = request_data_dict['model_type']

    initial_month = month
    initial_year = year

    closest_year, closest_month = get_max_year_month()

    if not month or not year:
        month = closest_month
        year = closest_year


    if month <= 1 or month >= 12:
        return JSONResponse(
            content={"error": "invalid month"},
            status_code=400
        )

    if year == 2020 and month < 11:
        return JSONResponse(
            content={"error": "too early for the prediction"},
            status_code=404
        )


    elif year == closest_month and month > closest_month:
        return JSONResponse(
            content={"error": "can't predict the future that far"},
            status_code=404
        )

    model = None

    if model_type == "BASE":
        model = baseline_model
    elif model_type == "TRAINED":
        model = trained_model

    else:
        return JSONResponse(
            content={"error": "invalid model type"},
            status_code=404
        )

    # for given user and given month
    features_to_predict, is_exist = get_features(user_id, month, year)

    if model_type == 'BASE':
        features_to_predict = {'year': year, 'month': month, 'user_id': user_id}

    if is_exist:
        prediction_price = model.predict(features_to_predict)

        json_result = {
            "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S:%f"),
            "user_id": user_id,
            "predicted_price": prediction_price
        }

        if initial_year == 2020 and initial_month == 12:
            json_result['year'] = 2021
            json_result['month'] = 1
        elif initial_year and initial_month:
            json_result['year'] = year
            json_result['month'] = month + 1
        else:
            json_result['year'] = year
            json_result['month'] = month

        return JSONResponse(
            content=json_result,
            status_code=200
        )

    else:
        return JSONResponse(
            content={"error": "can't make a prediction"},
            status_code=404
        )

# running application
if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=8000)
