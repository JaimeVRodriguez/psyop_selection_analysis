import pandas as pd
import matplotlib.pyplot as plt

def read_excel_files(filepath, filenames):
    dfs = []
    for i, filename in enumerate(filenames):
        file = filepath + '/' + filename
        dfs.append(pd.read_excel(file))
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined

def binary_language(df):
    df.LANG = (df.LANG.notna() & df.LANG.ne("")).astype(int)
    return df

def drop_na_values(df):
    df.dropna(how='any', inplace=True)
    return df

def flt_to_int(df, column):
    df[column] = df[column].astype(int)
    return df

def clean_data(filepath, filenames, columns):

    df = read_excel_files(filepath, filenames)
    df = binary_language(df)
    drop_na_values(df)
    for i in columns:
        df = flt_to_int(df, i)
    return df


def equal_column_val(df, column, value):
    result = df[df[column] == value]
    return result

def not_column_val(df, column, value):
    result = df[df[column] != value]
    return result

def bar_plot(data, labels, title, ylabel):
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_title(title, loc='left', size=18, fontweight='bold')
    ax.set_ylabel(ylabel)




if __name__ == '__main__':
    filepath = 'data/poas'
    filenames = ['poas16.xlsx', 'poas17.xlsx', 'poas18.xlsx', 'poas19.xlsx', 'poas20.xlsx', 'poas21.xlsx', 'poas22.xlsx']
    filepath2 = 'data/sfas'
    filenames2 = ['sfas16.xlsx', 'sfas17.xlsx', 'sfas18.xlsx', 'sfas19.xlsx', 'sfas20.xlsx', 'sfas21.xlsx', 'sfas22.xlsx']
    
    # poas = read_excel_files(filepath, filenames)
    # # poas = binary_language(poas)

    # poas = clean_data(filepath, filenames, ['RACE', 'SEC', 'AB', 'RGR', 'PT', 'DEP', 'AGE', 'EL', 'SC', 'CO', 'FA', 'ST' ], ['DEP', 'PT', 'AGE', 'EL', 'SC', 'CO', 'FA', 'ST'] )

    sfas = clean_data(filepath2, filenames2)
    print(sfas)



