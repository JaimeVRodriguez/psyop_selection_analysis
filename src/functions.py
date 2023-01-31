import pandas as pd
import matplotlib.pyplot as plt


def equal_column_val(df, column, value):
    result = df[df[column] == value]
    return result

def not_column_val(df, column, value):
    result = df[df[column] != value]
    return result

def annual_total(df):
    ser = df.CODE.groupby(df.YEAR).count()
    return ser







