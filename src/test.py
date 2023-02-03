import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def significant_features(df, target):
    '''This function takes in a DataFrame (df) and a target 
    column name (target). It creates two sets of data: X which 
    is all columns in the df except for the target column, and 
    y which is only the target column. The function then trains 
    a logistic regression model using the X data and the y data, 
    and uses the Recursive Feature Elimination (RFE) method to 
    determine the most significant features. RFE is used to select 
    the most important features by repeatedly fitting the model 
    and removing the least important feature until the desired number 
    of features is reached. The function returns the selected features 
    in the form of a list of column names.'''
    X = df.drop(target, axis=1)
    y = df[target]
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model)
    rfe = rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

def drop_unwanted_columns(df, cols):
    '''drop_unwanted_columns: This function takes in a DataFrame (df) 
    and a list of desired column names (cols). It first determines 
    which columns are unwanted, which are all columns in the df except 
    for the ones in the cols list. The function then drops the unwanted 
    columns from the DataFrame using the drop method and modifies the 
    original df in place. The function returns the modified df.'''
    unwanted_cols = set(df.columns) - set(cols)
    df.drop(unwanted_cols, axis=1, inplace=True)
    return df
