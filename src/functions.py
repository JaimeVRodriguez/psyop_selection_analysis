import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split

seed = 8

def equal_column_val(df, column, value):
    result = df[df[column] == value]
    return result

def not_column_val(df, column, value):
    result = df[df[column] != value]
    return result

def annual_total(df):
    ser = df.CODE.groupby(df.YEAR).count()
    return ser

def ttest(df1, df2, columns):
    tests = []
    for i in columns:
        val = stats.ttest_ind(df1[i], df2[i], equal_var=False)
        tests.append(val)
    return tests

def combined(df_list):
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df

def split_combined(df):
    df1 = df.loc[df.CODE == 0]
    df2 = df.loc[df.CODE == 1]
    return df1, df2


def pvalues(test_list):
    columns = ['GT', 'PT', 'AGE', 'LANG']
    pvals = {}
    for i, col in zip(test_list, columns):
        pvals[col] = i[1]
         
    return pvals 

def train_split(X, y, test_size=.2, seed=seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    return X_train, X_test, y_train, y_test





