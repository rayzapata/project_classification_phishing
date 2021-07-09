#Z0096


# import standard libraries
import pandas as pd
import numpy as np

from math import ceil
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#################### Prepare Data ####################


def drop_cols_null(df, max_missing_rows_pct=0.25):
    '''

    Takes in a DataFrame and a maximum percent for missing values and
    returns the passed DataFrame after removing any colums missing the
    defined max percent or more worth of rows

    '''
    
    # set threshold for axis=1 and drop cols
    thresh_col = ceil(df.shape[0] * (1 - max_missing_rows_pct))
    df = df.dropna(axis=1, thresh=thresh_col)

    return df


def drop_rows_null(df, max_missing_cols_pct=0.25):
    '''

    Takes in a DataFrame and a maximum percent for missing values and
    returns the passed DataFrame after removing any rows missing the
    defined max percent or more worth of columns

    '''
    
    # set threshold for axis=0 and drop rows
    thresh_row = ceil(df.shape[1] * (1 - max_missing_cols_pct))
    df = df.dropna(axis=0, thresh=thresh_row)
    
    return df


def drop_null_values(df, max_missing_rows_pct=0.25, max_missing_cols_pct=0.25):
    '''

    Takes in a DataFrame and maximum percents for missing values in
    columns and rows and returns the passed DataFrame after first
    removing any columns missing the defined max percent or more worth
    of rows then removing rows missing the defined max percent or more
    worth of columns


    '''
    
    # drop columns with null values for passed percent of rows
    df = drop_cols_null(df, max_missing_rows_pct)
    # drop rows with null values for passed percent of columns
    df = drop_rows_null(df, max_missing_cols_pct)
    
    return df


def shed_iqr_outliers(df, k=1.5, col_list=None):
    '''

    Takes in a DataFrame and optional column list and removes values 
    that are outside of the uppper and lower bounds for all columns or
    those passed within the list

    '''
    
    # if col_list=['list', 'of', 'cols'], apply outlier removal to cols
    # in col_list
    if col_list != None:
        # start loop for each column in col_list
        for col in col_list:
            # find q1 and q3
            q1, q3 = df[col].quantile([.25, .75])
            # calculate IQR
            iqr = q3 - q1
            # set upper and lower bounds
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            # return DataFrame with IQR outliers removed
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    # if col_list=None, apply outlier removal to all cols
    else:
        # start loop for each column in DataFrame
        for col in list(df):
            # find q1 and q3
            q1, q3 = df[col].quantile([.25, .75])
            # calculate IQR
            iqr = q3 - q1
            # set upper and lower bounds
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            # return DataFrame with IQR outliers removed
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df


def shed_zscore_outliers(df, z=3, col_list=None):
    '''

    Takes in a DataFrame and optional column list and removes values 
    that are three or more standard deviations from column mean for all
    columns or those passed within the list

    '''
    
    # if col_list=['list', 'of', 'cols'], apply outlier removal to cols
    # in col_list
    if col_list != None:
        # start loop for each column in col_list
        for col in col_list:
            # reassign DataFrame with column values only within 3
            # standard deviations of column mean
            df = df[np.abs(zscore(df[col])) < z]
    # if col_list=None, apply outlier removal to all cols
    else:
        # start loop for each column in DataFrame
        for col in list(df):
            # reassign DataFrame with column values only within 3
            # standard deviations of column mean
            df = df[np.abs(zscore(df[col])) < z]

    return df


def split_data(df, stratify=None):
    '''

    Takes in a DataFrame and splits it into 60%/20%/20% for train,
    validate, and test DataFrames using random_rate=19

    '''

    if stratify == None:
        # split data into train, validate, and test datasets
        train_validate, test = train_test_split(df, test_size=0.2, 
                                                random_state=19)
        train, validate = train_test_split(train_validate, test_size=0.25,
                                                random_state=19)
    else:
        # split data into train, validate, and test datasets
        train_validate, test = train_test_split(df, test_size=0.2, 
                                                random_state=19, stratify=df[stratify])
        train, validate = train_test_split(train_validate, test_size=0.25,
                                                random_state=19, stratify=train_validate[stratify])

    return train, validate, test


def split_xy(train, validate, test, target):
    '''

    Takes in the three train, validate, and test DataFrames and returns
    six X, y DataFrames after splitting the target from the X data

    '''

    # split train into X, y
    X_train = train.drop(columns=target)
    y_train = pd.DataFrame(train[target])
    # split validate into X, y
    X_validate = validate.drop(columns=target)
    y_validate = pd.DataFrame(validate[target])
    # split test into X, y
    X_test = test.drop(columns=target)
    y_test = pd.DataFrame(test[target])

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def impute_null_values(train, validate, test, strategy='mean', col_list=None):
    '''

    Takes in the train, validate, and test DataFrame and imputes either
    all columns the passed column list with the strategy defined in 
    arguments

    strategy='mean' default behavior

    '''

    # if no list is passed, impute all values
    if col_list != None:
        for col in col_list:
            imputer = SimpleImputer(strategy=strategy)
            train[[col]] = imputer.fit_transform(train[[col]])
            validate[[col]] = imputer.transform(validate[[col]])
            test[[col]] = imputer.transform(test[[col]])
    # if col_list is passed, impute only values within
    else:
        for col in list(train):
            imputer = SimpleImputer(strategy=strategy)
            train[[col]] = imputer.fit_transform(train[[col]])
            validate[[col]] = imputer.transform(validate[[col]])
            test[[col]] = imputer.transform(test[[col]])

    return train, validate, test
