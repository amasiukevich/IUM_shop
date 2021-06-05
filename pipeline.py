import os
import joblib

from transform import transform_all_data
from constants import DATA_PART_TRAINING, BASE_DIR

from training.feature_extractor import extract_features

from training.model_trainer import train_best_model


### training

# gathering new raw data

# Raw jsonl's

# transform
transform_all_data(
    f'{os.sep}raw_data{os.sep}data_version{DATA_PART_TRAINING}',
    f'{os.sep}proper_data{os.sep}data_version{DATA_PART_TRAINING}'
)

# extract features
feature_dataframe = extract_features(f'proper_data{os.sep}data_version{DATA_PART_TRAINING}')

# train a model
model, model_name = train_best_model(feature_dataframe)

# save a model
os.chdir(BASE_DIR)
joblib.dump(model, os.getcwd() + f'{os.sep}trained_models{os.sep}{model_name}.pkl')


### retrain