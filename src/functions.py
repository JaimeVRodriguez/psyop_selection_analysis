import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


def equal_column_val(df, column, value):
    '''This function takes a dataframe df, a column name column, 
    and a value value as inputs. It returns a new dataframe that 
    contains only the rows from the original dataframe where the 
    value in the specified column is equal to the input value.'''
    result = df[df[column] == value]
    return result

def not_column_val(df, column, value):
    '''This function takes a dataframe df, a column name column, 
    and a value value as inputs. It returns a new dataframe that 
    contains only the rows from the original dataframe where the 
    value in the specified column is not equal to the input value.'''
    result = df[df[column] != value]
    return result

def annual_total(df):
    '''This function takes a dataframe df as input and returns a pandas 
    series that groups the CODE column by the YEAR column and counts the 
    number of unique values in each group.'''
    ser = df.CODE.groupby(df.YEAR).count()
    return ser

def ttest(df1, df2, columns):
    '''This function takes two dataframes df1 and df2 and a list of column 
    names columns as inputs. It performs a two-sample independent t-test for 
    each column in the list and returns a list of t-test results.'''
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

def new_labels(df, new_labels):
    df.rename(columns=new_labels, inplace=True)
    return df

def replace_vals(df, column, new_vals):
    df[column] = df[column].replace(new_vals)

def return_x_y(df, target):
    X = df.drop(target, axis=1)
    X2 = Normalizer().fit_transform(X)
    y = df[target]
    return X, X2, y




