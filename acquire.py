#Z0096

# import standard libraries
import pandas as pd
import numpy as np

# import file checker
from os.path import isfile

# import created function
from prepare import split_data


#################### Acquire Data ####################


def get_csv(path):
    '''

    Reads in CSV file to pandas DataFrame, drops duplicates, and
    repalces whitespace values with np.NaN values

    '''

    # acquire data from cSV file
    df = pd.read_csv(path)
    # drop any existing duplicates
    df = df.drop_duplicates()
    # replace whitespace with nan values
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    return df


def wrangle_phishing():
    '''

    For phishing data, uses get_csv function and returns a DataFrame
    with columns with URL related data and renames to a more more
    easily parsed format to improve data clarity. Returns data split
    into train, validate, and test data sets
    
    '''

    # get DataFrame from CSV file
    df = get_csv('Phishing_Legitimate_full.csv')
    # create variable to hold list of URL specific columns
    col_list = [
                'NumDots', 'SubdomainLevel', 'PathLevel',
                'UrlLength', 'NumDash', 'NumDashInHostname',
                'AtSymbol', 'TildeSymbol', 'NumUnderscore',
                'NumPercent', 'NumQueryComponents', 'NumAmpersand',
                'NumHash', 'NumNumericChars',
                'RandomString', 'DomainInSubdomains', 'DomainInPaths',
                'HttpsInHostname', 'HostnameLength', 'PathLength',
                'QueryLength', 'DoubleSlashInPath', 'NumSensitiveWords',
                'CLASS_LABEL']
    # create dictionary for column name changes
    rename_dict = {
                'NumDots':'num_dot_url',
                'SubdomainLevel':'subdomain_level',
                'PathLevel':'path_level',
                'UrlLength':'url_char_length',
                'NumDash':'num_dash_url',
                'NumDashInHostname':'num_dash_hostname',
                'AtSymbol':'has_at_symbol',
                'TildeSymbol':'has_tilde',
                'NumUnderscore':'num_underscore_url',
                'NumPercent':'num_percent_sign',
                'NumQueryComponents':'num_queries',
                'NumAmpersand':'num_ampersand',
                'NumHash':'num_hash',
                'NumNumericChars':'num_numerics',
                'RandomString':'has_random_string',
                'DomainInSubdomains':'domain_in_subdomain',
                'DomainInPaths':'domain_in_path',
                'HttpsInHostname':'https_in_hostname',
                'HostnameLength':'hostname_length',
                'PathLength':'path_length',
                'QueryLength':'query_length',
                'DoubleSlashInPath':'doubleslash_in_path',
                'NumSensitiveWords':'num_sensitive_words',
                'CLASS_LABEL':'is_phishing_attempt'}
    # filter to desired columns and rename
    df = df[col_list]
    df = df.rename(columns=rename_dict)
    # drop columns with no use
    df = df.drop(columns=['https_in_hostname', 'has_at_symbol',
                            'num_hash', 'doubleslash_in_path'])
    # sort columns alphabetically
    cols = df.columns.to_list()
    cols.sort()
    df = df[cols]
    # split data into train, validate, test
    train, validate, test = split_data(df, stratify='is_phishing_attempt')

    return train, validate, test


#################### Summarize Data ####################


def summarize(df):
    '''
    
    Takes in a single argument as a pandas DataFrame and outputs
    several statistics on passed DataFrame
    
    '''
    
    print('\n--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--\n')
    print('*** First Three Observations of DataFrame\n')
    print(df.head(3).T)
    print('\n--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--\n')
    print ('*** DataFame .info()\n')
    print(df.info())
    print('\n--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--\n')
    print('*** DataFrame .describe().T\n')
    print (df.describe().T)
    print('\n--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--\n')
    print('*** Value Counts for DataFrame Columns\n')
    cat_cols = [col for col in list(df) if df[col].dtype == 'O']
    num_cols = [col for col in list(df) if df[col].dtype != 'O']
    for col in list(df):
        if col in cat_cols:
            print(f'+ {df[col].name}\n\n{df[col].value_counts()}\n')
        else:
            print(f'+ {df[col].name}\n\n{df[col].value_counts(bins=10, sort=False)}\n')
