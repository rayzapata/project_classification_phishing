#Z0096


# import standard libraries
import pandas as pd
import numpy as np

from math import ceil
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#################### Prepare Data ####################


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

