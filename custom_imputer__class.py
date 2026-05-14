from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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