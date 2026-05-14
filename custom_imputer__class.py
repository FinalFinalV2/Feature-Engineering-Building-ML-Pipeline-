from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
class HorsepowerEngineRatio(BaseEstimator, TransformerMixin):
    
    def __init__(self, hp_col="Horsepower", engine_col="Engine_Size"):
        self.hp_col = hp_col
        self.engine_col = engine_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df["HP_to_Engine_Ratio"] = df[self.hp_col] / (df[self.engine_col] + 0.001)
        return df
    def set_output(self, transform = "pandas"):
        return self
    
class HorsepowerImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='Brand', target_col='Horsepower'):
        self.group_col = group_col
        self.target_col = target_col
        self.median_map = None

    def fit(self, X, y=None):
        self.median_map = X.groupby(self.group_col)[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.target_col] = X[self.target_col].fillna(X[self.group_col].map(self.median_map))
        
        return X[[self.target_col]]
    
    def set_output(self, transform = "pandas"):
        return self
    
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        if 'Model_Year' in X.columns:
            X['Model_Year'] = X['Model_Year'].mask(X["Model_Year"] < 1900, np.nan)

        if 'Mileage' in X.columns:
            X['Mileage'] = X['Mileage'].mask(X["Mileage"] > 800000, np.nan)
        
        if 'Horsepower' in X.columns:
            X['Horsepower'] = X['Horsepower'].mask(X["Horsepower"] > 1500, np.nan)

        if 'Engine_Size' in X.columns:
            X["Engine_Size"] = X["Engine_Size"].astype(str).str.replace(' cc', '', case=False)
            X["Engine_Size"] = pd.to_numeric(X["Engine_Size"], errors='coerce')
            X["Engine_Size"] = X["Engine_Size"].mask(X["Engine_Size"] > 10, X["Engine_Size"] / 1000)

        if 'Fuel_Type' in X.columns:
            X['Fuel_Type'] = X['Fuel_Type'].astype(str).str.lower().str.strip()
            
            if 'Brand' in X.columns:
                X.loc[X['Brand'] == 'Tesla', 'Fuel_Type'] = 'electric'

        return X
    def set_output(self, transform = "pandas"):
        return self