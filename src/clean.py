import pandas as pd
import matplotlib.pyplot as plt

def read_excel_files(filepath, filenames):
    '''
    Parameters:
    filepath: The file path where the Excel files are located.
    filenames: A list of filenames for the Excel files to be read in.
    
    This funciton returns single DataFrame that is created by 
    concatenating the data from the input Excel files.'''
    dfs = []
    for i, filename in enumerate(filenames):
        file = filepath + '/' + filename
        dfs.append(pd.read_excel(file))
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined

def binary_language(df):
    '''
    Parameters:
    `df`: DataFrame that will be modified in-place by the function

    The function modifies the DataFrame by creating a new column 
    called "LANG", which is filled with binary values (either 0 or 1).'''
    df.LANG = (df.LANG.notna() & df.LANG.ne("")).astype(int)
    return df

def binary_result(df):
    '''
    Parameters:
    `df`: DataFrame that will be modified in-place by the function

    The function modifies the DataFrame by creating a new column called 
    "CODE", which is filled with binary values (either 0 or 1). '''
    val_to_replace = ['SELECTED']
    df.CODE = df.RESULT.apply(lambda x: 1 if x in val_to_replace else 0)
    return df

def binary_TF(df):
    '''
    Parameters:
    `df`: DataFrame that will be modified in-place by the function

    The function modifies the DataFrame by replacing the values "F" and 
    "T" with binary values (0 and 1 respectively) in-place.'''
    df.replace({'F': 0, 'T': 1}, inplace=True)
    return df

def drop_na_values(df):
    '''
    Parameters:
    `df`: DataFrame that will be modified in-place by the function

    The function modifies the DataFrame by dropping all rows that contain 
    at least one missing value, in-place.'''
    df.dropna(how='any', inplace=True)
    return df

def flt_to_int(df, column):
    '''
    Parameters:
    `df`: DataFrame that will be modified in-place by the function
    `column`: The name of the column in the DataFrame that will be 
    converted from float to integer type

    The function modifies the DataFrame by converting the values in the 
    specified column from float to integer type, in-place.'''
    df[column] = df[column].astype(int)
    return df

def clean_data(filepath, filenames, columns, group):
    '''
    Parameters:
    `filepath`: The file path where the excel files are located.
    `filenames`: The list of file names of the excel files to be read 
    and concatenated.
    `columns`: The list of column names in the DataFrame that will be 
    converted from float to integer type.
    `group`: The value that will be used to fill the "GROUP" column in 
    the final DataFrame.
    
    The function reads multiple excel files located at the specified filepath 
    and with the specified names in the filenames list, and concatenates them 
    into a single Pandas DataFrame df.'''
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