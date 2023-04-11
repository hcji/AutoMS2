# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:23:35 2023

@author: gls
"""


'''
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes


rng = np.random.RandomState(42)

X_diabetes, y_diabetes = load_diabetes(return_X_y=True)
X_diabetes = X_diabetes[:300]
y_diabetes = y_diabetes[:300]


def add_missing_values(X_full, y_full):
    n_samples, n_features = X_full.shape

    # Add missing values in 75% of the lines
    missing_rate = 0.75
    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=bool)
    missing_samples[:n_missing_samples] = True

    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[missing_samples, missing_features] = np.nan
    y_missing = y_full.copy()

    return X_missing, y_missing


X_miss_diabetes, y_miss_diabetes = add_missing_values(X_diabetes, y_diabetes)

imputer = Imputer(X_miss_diabetes, y_miss_diabetes)
imputer.fill_with_low_value()
imputer.fill_with_mean_value()
imputer.fill_with_median_value()
imputer.fill_with_knn_imputer()
imputer.fill_with_iterative_RF()
imputer.fill_with_iterative_BR()
imputer.fill_with_iterative_SVR()
'''


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

class Imputer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
        
    def fill_with_low_value(self):
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=10**-6, keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y
    
    
    def fill_with_mean_value(self):
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y
    
    
    def fill_with_median_value(self):
        imputer = SimpleImputer(missing_values=np.nan, strategy="median", keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y        
    
    
    def fill_with_knn_imputer(self, n_neighbors = 3):
        imputer = KNNImputer(missing_values=np.nan, n_neighbors = n_neighbors, keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y 


    def fill_with_iterative_RF(self, n_estimators = 100, max_depth = 2, max_iter = 100):
        estimator = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        imputer = IterativeImputer(estimator=estimator, missing_values=np.nan,max_iter=max_iter, keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y     
    
    
    def fill_with_iterative_BR(self, max_iter = 100):
        imputer = IterativeImputer(missing_values=np.nan, max_iter=max_iter, keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y         
        
    
    def fill_with_iterative_SVR(self, max_iter = 100):
        estimator = SVR()
        imputer = IterativeImputer(estimator=estimator, missing_values=np.nan,max_iter=max_iter, keep_empty_features=True)
        x = imputer.fit_transform(self.x)
        y = self.y
        return x, y        
        