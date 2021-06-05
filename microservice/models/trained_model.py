import os
import joblib
from constants import BASE_DIR

class TrainedModel():

    def __init__(self):

        os.chdir(BASE_DIR + f"{os.sep}trained_models{os.sep}")
        self.restoring_filename = 'random_forest.pkl'
        try:
            self.model = joblib.load(self.restoring_filename)
        except:
            print("Can't restore a model")


    def predict(self, single_item):
        return self.model.predict(single_item)[0]