"""Data processing and loading modules""" 

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Optional
import yaml

class RealEstateDataLoader:
    def __init__(self, config_path: str="configs/data_config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)['data'] # load the data config
        self.raw_data_path = self.config['raw_data_path'];
    
    def load_apartments_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.raw_data_path, self.config['apartments_file']) # join the raw data path with the apartments file
        print(f"Loading apartments data from {file_path}")
        try:
            df = pd.read_csv(file_path, skiprows = 1, header = 0, delimiter = ';') # skip the first row containing headers
            print(f"Loaded dataframe : {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"File not found at {file_path}")
            raise 
        except Exception as e:
            print(f"Error loading apartments data: {e}")
            raise
    
    def load_localities_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.raw_data_path, self.config['localities_file']) 
        print(f"Loading localities data from {file_path}")
        try:
            df = pd.read_csv(file_path, skiprows = 1, header = 0)
            print(f"Loaded dataframe : {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"File not found at {file_path}")
            raise
        except Exception as e:
            print(f"Error loading localities data: {e}")
            raise
    
    def load_ancpi_trends(self) -> pd.DataFrame: 
        file_path = os.path.join(self.raw_data_path, self.config['ancpi_trends_file']) 
        print(f"Loading ANCPI trends data from {file_path}")
        try:
            df = pd.read_csv(file_path, skiprows = 1, header = 0)
            print(f"Loaded dataframe : {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"File not found at {file_path}")
            raise
        except Exception as e:
            print(f"Error loading ANCPI trends data: {e}")
            raise
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        datasets = {
            'apartments': self.load_apartments_data(),
            'localities': self.load_localities_data(),
            'ancpi_trends': self.load_ancpi_trends()
        }
        return datasets
    

def main():
    loader = RealEstateDataLoader()
    datasets = loader.load_all_data()
    print(datasets['apartments'].head())
    print(datasets['localities'].head())
    print(datasets['ancpi_trends'].head())

if __name__ == "__main__":
    main()
    
