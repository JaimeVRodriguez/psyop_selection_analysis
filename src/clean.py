import pandas as pd
import matplotlib.pyplot as plt

def read_excel_files(filepath, filenames):
    '''
    Parameters
    filepath (str): The file path where the Excel files are located.
    filenames (list of str): A list of filenames for the Excel files to be read in.
    
    This funciton returns single DataFrame that is created by concatenating the data from the input Excel files.'''
    dfs = []
    for i, filename in enumerate(filenames):
        file = filepath + '/' + filename
        dfs.append(pd.read_excel(file))
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined

def binary_language(df):
    df.LANG = (df.LANG.notna() & df.LANG.ne("")).astype(int)
    return df

def binary_result(df):
    val_to_replace = ['SELECTED']
    df.CODE = df.RESULT.apply(lambda x: 1 if x in val_to_replace else 0)
    return df

def binary_TF(df):
    df.replace({'F': 0, 'T': 1}, inplace=True)
    return df

def drop_na_values(df):
    df.dropna(how='any', inplace=True)
    return df

def flt_to_int(df, column):
    df[column] = df[column].astype(int)
    return df

def clean_data(filepath, filenames, columns, group):
    df = read_excel_files(filepath, filenames)
    df = binary_language(df)
    df = binary_result(df)
    df = binary_TF(df)
    drop_na_values(df)
    for i in columns:
        df = flt_to_int(df, i)
    df['GROUP'] = group
    return df

if __name__ == '__main__':
    filepath = 'data/poas'
    filenames = ['poas16.xlsx', 'poas17.xlsx', 'poas18.xlsx', 'poas19.xlsx', 'poas20.xlsx', 'poas21.xlsx', 'poas22.xlsx']
    filepath2 = 'data/sfas'
    filenames2 = ['sfas16.xlsx', 'sfas17.xlsx', 'sfas18.xlsx', 'sfas19.xlsx', 'sfas20.xlsx', 'sfas21.xlsx', 'sfas22.xlsx']
    columns = ['DEP', 'PT', 'AGE', 'GT', 'EL', 'SC', 'CO', 'FA', 'ST']
    # poas = read_excel_files(filepath, filenames)
    # # poas = binary_language(poas)

    # poas = clean_data(filepath, filenames, ['RACE', 'SEC', 'AB', 'RGR', 'PT', 'DEP', 'AGE', 'EL', 'SC', 'CO', 'FA', 'ST' ], ['DEP', 'PT', 'AGE', 'EL', 'SC', 'CO', 'FA', 'ST'] )

    sfas = clean_data(filepath2, filenames2, columns, 'SFAS')
    print(sfas)