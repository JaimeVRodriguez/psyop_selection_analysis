import pandas as pd
import scipy.stats as stats


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
    df1 = df.loc[df.CODE == 1]
    df2 = df.loc[df.CODE == 0]
    return df1, df2


def pvalues(test_list):
    columns = ['GT', 'PT', 'AGE', 'LANG']
    pvals = {}
    for i, col in zip(test_list, columns):
        pvals[col] = i[1]
         
    return pvals 





