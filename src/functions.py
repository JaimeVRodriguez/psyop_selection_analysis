import pandas as pd

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



if __name__ == '__main__':
    filepath = 'data/poas'
    filenames = ['poas16.xlsx', 'poas17.xlsx', 'poas18.xlsx', 'poas19.xlsx', 'poas20.xlsx', 'poas21.xlsx', 'poas22.xlsx']

    poas = read_excel_files(filepath, filenames)

    print(poas)



