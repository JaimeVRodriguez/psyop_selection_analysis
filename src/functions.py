import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

seed = 16486415


def equal_column_val(df, column, value):
    '''
    Parameters:
    `df`: The dataframe to be filtered.
    `column`: The name of the column to be used for filtering.
    `value`: The value to be used for filtering.
  
    The function returns a dataframe that contains only the rows where 
    the specified column has the specified value.'''
    result = df[df[column] == value]
    return result

def not_column_val(df, column, value):
    '''
    Parameters:
    `df`: The input dataframe.
    `column`: The name of the column in df to be evaluated for equality with value.
    `value`: The value to be compared to the values in the specified column.

    Returns a new dataframe (pd.DataFrame) that includes only the rows from the input 
    dataframe df where the value in column is not equal to value.'''
    result = df[df[column] != value]
    return result

def annual_total(df):
    '''
    Parameters:
    `df`: The input dataframe for which the total number of occurrences 
    for each year in the 'CODE' column needs to be calculated.
    
    This function calculates the total number of occurrences for each year in the 'CODE' 
    column of the input dataframe 'df'.'''
    ser = df.CODE.groupby(df.YEAR).count()
    return ser

def ttest(df1, df2, columns):
    '''
    Parameters:
    `df1`: The first dataframe to be tested.
    `df2`: The second dataframe to be tested.
    `columns`: A list of column names to be tested in the two dataframes.

    The function returns a list of t-test results, where each result is a tuple containing 
    the t-statistic and the corresponding p-value. The p-value is used to determine the 
    significance of the difference between the means, with a lower p-value indicating a 
    stronger evidence against the null hypothesis (i.e. the means are significantly different).'''
    tests = []
    for i in columns:
        val = stats.ttest_ind(df1[i], df2[i], equal_var=False)
        tests.append(val)
    return tests

def combined_df(df_list):
    '''
    Parameters:
    `df_list`: A list of dataframes to be concatenated.

    The function takes a list of dataframes, df_list, as its input. It concatenates the dataframes 
    along the rows (axis=0) and resets the index to start from'''
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df

def split_combined(df):
    '''
    Parameters:
    `df`: pandas dataframe, the dataframe to be split into two separate dataframes.

    The function takes in a pandas dataframe and splits it into two separate dataframes 
    based on the value of the "CODE" column. df1 contains rows where the "CODE" column is 
    equal to 0. df2 contains rows where the "CODE" column is equal to 1. Both dataframes are 
    returned as a tuple.'''
    df1 = df.loc[df.CODE == 0]
    df2 = df.loc[df.CODE == 1]
    return df1, df2


def pvalues(test_list):
    '''
    Parameters:
    `test_list`: a list of tuples, where each tuple contains a test result in the form of (test_name, p-value).
    
    The function takes in a list of test results as a list of tuples and extracts the p-values from the tuples.
    The list of column names is defined as columns = ['GT', 'PT', 'AGE', 'LANG']. The p-values are stored in a 
    dictionary, pvals, where the keys are the column names and the values are the corresponding p-values. The
    resulting dictionary is returned.'''
    columns = ['GT', 'PT', 'AGE', 'LANG']
    pvals = {}
    for i, col in zip(test_list, columns):
        pvals[col] = i[1]
    return pvals 

def train_split(X, y, test_size=.2, seed=seed):
    '''
    Parameters:
    `X`: a 2-dimensional numpy array or pandas dataframe, representing the features of the 
    data to be split.
    `y`: a 1-dimensional numpy array or pandas series, representing the target values.
    `test_size`: float, the proportion of the data to be allocated to the test set 
    (default value is .2).
    `seed`: an integer value, used as the random seed for the splitting process 
    (default value is None).

    The function takes in the feature data X and target data y, and splits them into training 
    and testing sets. The split is performed using the train_test_split function from the scikit-learn 
    library with test_size argument specified.The stratify argument is set to y to ensure the class 
    distribution in the target data is maintained in both the training and testing sets. The random_state 
    argument is set to the specified seed value to ensure reproducibility of the split.The resulting 
    training and testing sets for both the features and the target values are returned as a tuple.'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    return X_train, X_test, y_train, y_test

def new_labels(df, new_labels):
    '''
    Parameters:
    df: A pandas dataframe which needs to be modified.
    new_labels: A dictionary with old labels as keys and new labels as values, used to rename the columns 
    of the dataframe df.

    The function modifies the dataframe df in place by renaming the columns using the dictionary new_labels. 
    The modified dataframe is then returned as the output of the function.'''
    df.rename(columns=new_labels, inplace=True)
    return df

def replace_vals(df, column, new_vals):
    '''
    Parameters:
    `df`: A pandas dataframe which needs to be modified.
    `column`: A string representing the name of the column in the dataframe df to be modified.
    `new_vals`: A dictionary with old values as keys and new values as values, used to replace 
    the values in the specified column of the dataframe df.

    The function modifies the dataframe df by replacing the values in the specified column 
    column using the dictionary new_vals. The modified dataframe is not returned as the output 
    of the function and the changes are made in place.'''
    df[column] = df[column].replace(new_vals)

def return_x_y(df, target):
    '''
    Parameters:
    `df`: A pandas dataframe which needs to be processed.
    `target`: A string representing the name of the target column in the dataframe df.

    The function returns three outputs:
    X: A dataframe which is a modified version of the input dataframe df with the target column target removed.
    X2: A normalized version of the dataframe X using the Normalizer class from scikit-learn.
    y: A pandas series representing the values of the target column target from the input dataframe df.'''
    X = df.drop(target, axis=1)
    X2 = Normalizer().fit_transform(X)
    y = df[target]
    return X, X2, y

def significant_features(df, target):
    '''
    Parameters
    `df`: A pandas dataframe which needs to be processed.
    `target`: A string representing the name of the target column in the dataframe df.

    The function returns a list of strings representing the selected features from the input dataframe df.'''
    X = df.drop(target, axis=1)
    y = df[target]
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model)
    rfe = rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

def drop_unwanted_columns(df, cols):
    '''
    Parameters:
    `df`: A pandas dataframe which needs to be modified.
    `cols`: A list of strings representing the columns in the dataframe df that should be kept.

    The function modifies the dataframe df in place by removing columns that are not present 
    in the list cols. The modified dataframe is then returned as the output of the function.'''
    unwanted_cols = set(df.columns) - set(cols)
    df.drop(unwanted_cols, axis=1, inplace=True)
    return df

def sig_feature_split(df, target):
    '''
    Parameters:
    `df`: A pandas dataframe which needs to be processed.
    `target`: A string representing the name of the target column in the dataframe df.

    The function returns four dataframes: X_test, X_train, y_test, and y_train.'''
    sig_features = list(significant_features(df, target))
    sig_features.append(target)
    refined_df = drop_unwanted_columns(df, sig_features)
    X, X2, y = return_x_y(refined_df, target)
    X_test, X_train, y_test, y_train = train_split(X2, y)
    return X_test, X_train, y_test, y_train

def outcomes(df, column, value):
    '''
    Parameters:
    `df`: A pandas dataframe which needs to be processed.
    `column`: A string representing the name of the column in the dataframe df to be used to determine the outcome.
    `value`: A scalar value to be used to split the dataframe df into two parts.

    The function returns a list split and two dataframes selected and not_selected.'''
    selected = equal_column_val(df, column, value)
    not_selected = not_column_val(df, column, value)
    split = [selected[column].count(), not_selected[column].count()]
    return split, selected, not_selected

def age_counts(df1, df2, column):
    '''
    Parameters:
    `df1`: First dataframe to be grouped
    `df2`: Second dataframe to be grouped
    `column`: Column to group the dataframes by

    This function groups the data in `df1` and `df2` by the specified `column`,
    and returns two resulting dataframes with the count of each group.
    '''
    ages1 = df1.groupby(column).size().reset_index(name='counts')
    ages2 = df2.groupby(column).size().reset_index(name='counts')
    return ages1, ages2

def combined_features(df1, df2, column, value):
    '''
    Parameters:
    `df1`: First dataframe to be combined
    `df2`: Second dataframe to be combined
    `column`: Column to select the rows by
    `value`: Value that the `column` should have
    
    This function combines `df1` and `df2` into a single dataframe and returns 
    the resulting dataframe and the rows where the specified `column` has the 
    specified `value`.'''
    both = [df1, df2]
    combined = combined_df(both)
    combined_selected = equal_column_val(combined, column, value)
    return combined, combined_selected



