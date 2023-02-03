import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def get_most_significant_features(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model)
    rfe = rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    
    return selected_features



def drop_unwanted_columns(df, cols):
    unwanted_cols = set(df.columns) - set(cols)
    df.drop(unwanted_cols, axis=1, inplace=True)
    return df
