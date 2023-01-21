import dill
import pandas as pd
from pathlib import Path


class DataPreparation:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()

    def run(self):
        nlp_df = self.data
        return 1+1

    def read_data(self):
        file_name = Path(self.config.data_path)
        return pd.read_pickle(file_name.resolve())
