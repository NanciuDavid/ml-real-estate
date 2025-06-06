"Data cleaning and preprocessing Utilities"
import data_loader
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import yaml

class RealEstateDataCleaner:
    def __init__(self, config_path: str="configs/data_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)['data']
        self.target_col = self.config['target_column']
        self.columns_to_drop = self.config['columns_to_drop']
        
    
    def replace_nan_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #Replacing the 'fara informatii values' with NaN
        nan_strings = ['fără informații', 'n/a', 'fara informatii', ' ', '']
        
        def replace_value(value):
            if isinstance(value, str) and value.lower() in nan_strings:
                return np.nan
            return value
        
        for col in df.columns:
            df[col] = df[col].apply(replace_value)
            
        return df

    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #cleaning and converting numeric columns
        numeric_cols = {
            'floor' : {
                'mappings' : {
                    'parter' : 0, 'demisol' : -1, 'parter înalt' : 0.5
                    }
                },
            'surface' : {},
            'price_per_sqm' : {},
            'rooms' : {},
            'construction_year' : {},
            'price' : {}, # target col
            }
        
        # rest of the columns like distances, accesibility score, will be lift as they were computed
        for col, settings in numeric_cols.items():
            if col in df.columns:
                if 'mappings' in settings:
                    df[col] = df[col].replace(settings['mappings'])
                
                df[col] = pd.to_numeric(df[col], errors='coerce') # convert to numeric, if error, convert to NaN
                print(f" {col}: {df[col].notna().sum()} / {len(df)} valid values")
                
        return df
    
    def handle_outlier(self, df: pd.DataFrame, method: str = 'iqr'):
        
        def mask_iqr(col_series: pd.Series) -> pd.Series:
            """
            Return a Boolean Series: True if the value is within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            """
            q1 = col_series.quantile(0.25)
            q3 = col_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (col_series >= lower_bound) & (col_series <= upper_bound)
    
        numeric_cols = ['floor', 'surface', 'price_per_sqm', 'rooms', 'price']
        df_clean = df.copy()
        
        # Iterate column by column, drop rows flagged as outliers
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue  # skip columns that aren’t present

            # Skip columns that are non‐numeric or all‐NaN
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            if method.lower() == "iqr":
                inlier_mask = mask_iqr(df_clean[col]) 
            
            df_clean = df_clean[inlier_mask] 
        return df_clean
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.replace_nan_strings(df)
        df = self.clean_numeric_columns(df)
        df = self.handle_outlier(df, method = 'iqr')
        
        columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        
        if columns_to_drop:
            df = df.drop(columns = columns_to_drop)
            print(f"Dropped columns : {columns_to_drop}")
            
        return df
    

def main():
    cleaner = RealEstateDataCleaner()
    loader = data_loader.RealEstateDataLoader()
    datasets = loader.load_all_data()
    df_clean = cleaner.clean_data(datasets['apartments'])
    print(df_clean.head())
    
if __name__ == "__main__":
    main()
    
    
    