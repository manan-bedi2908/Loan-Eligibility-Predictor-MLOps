import os
import pandas as pd
import joblib
from prediction_model.config import config

def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

