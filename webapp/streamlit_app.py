#Z0096


# import standard libraries
import pandas as pd
import numpy as np

# import create functions
import sys, os
sys.path.append(os.path.abspath("../"))
from acquire import wrangle_phishing
from prepare import split_xy
from explore import create_clusters

# import parse, scaler, model
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# import streamlit
import streamlit as st
import streamlit.components.v1 as components
from bokeh.models.widgets import Div


#################### Fit Data ####################

def fit_data():
    '''
    '''

    # read data into DataFrames
    train, validate, test = wrangle_phishing()
    # create clusters to get kmeans object
    train, validate, test, kmeans = create_clusters(train, validate, test,
                              ['num_dot_url', 'num_dash_url'], 'dash_dot', k=4)
    X_train, y_train, _, _, _, _ = split_xy(train, validate, test,
                                            'is_phishing_attempt')
    # filter DataFrame to only require columns
    X_train = X_train[['num_dash_url',
                      'num_dot_url',
                      'num_numerics',
                      'num_queries',
                      'num_sensitive_words',
                      'path_length',
                      'path_level',
                      'query_length',
                      'url_char_length',
                      'dash_dot_clstr_2']]
    # scale data for modeling
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    # model data for prediction algorithm
    # create and fit RF model with kbest_ten features
    model = RandomForestClassifier(n_estimators=500, max_depth=10,
                                   bootstrap=True, random_state=19)
    model.fit(X_train, y_train.is_phishing_attempt)

    return scaler, kmeans, model


#################### Parse URL ####################


def url_parse():
    '''
    '''
    
    # request input of URL
    url = st.text_input('Enter URL')
    # parse needed information from url and assign to variabls
    _, hostname, path, _, query, _ = urlparse(url)
    
    return url, hostname, path, query


def get_num_dash_url(url):
    '''
    '''
    
    # create loop to count dashes in URL
    count = 0
    for char in url:
        if char == '-':
            count += 1
    num_dash_url = count
    
    return num_dash_url


def get_num_dot_url(url):
    '''
    '''
    
    # create loop to count dots in URL
    count = 0
    for char in url:
        if char == '.':
            count += 1
    num_dot_url = count
    
    return num_dot_url


def get_num_numerics(url):
    '''
    '''
    
    # num_numerics
    count = 0
    for char in url:
        if char.isnumeric() == True:
            count += 1
    num_numerics = count
    
    return num_numerics


def get_num_queries(query):
    '''
    '''
    
    # split queries on question mark and get list for counting
    query_list = query.split(sep='?')
    num_queries = len(query_list)
    
    return num_queries


def get_num_sensitive_words(hostname, path, query):
    '''
    '''
    
    # create list of senstive words
    word_list = ['secure', 'account', 'login', 'password', 'api',
                 'signin', 'banking', 'secured', 'safe', 'webscr',
                 'logon', 'inloggen', 'register', 'webhost', 'pay',
                 'payment', 'dollar', 'shop', 'signon', 'net-acc', 
                 'acc', 'finance', 'dns', 'host']
    # set zero for initial count
    count = 0
    # split path into format to count words
    sub_host = hostname.split(sep=(r'./1234567890`~!@#$%^&*()-_=+;:,<>\|\'"'),
                                                             maxsplit=50)   
    # create loop to count occurence of senstive words in hostname
    for sub in sub_host:
        for word in word_list:
            if sub.lower().__contains__(word):
                count+= 1
    # split path into format to count words
    sub_path = path.split(sep=(r'./1234567890`~!@#$%^&*()-_=+;:,<>\|\'"'),
                                                             maxsplit=50)   
    # create loop to count occurence of sensitive words in path
    for sub in sub_path:
        for word in word_list:
            if sub.lower().__contains__(word):
                count+= 1
    # split query into format to count words
    sub_query = query.split(sep=(r'./1234567890`~!@#$%^&*()-_=+;:,<>\|\'"'),
                                                             maxsplit=50)
    # create loop to count occurence of sensitive words in queries
    for sub in sub_query:
        for word in word_list:
            if sub.lower().__contains__(word):
                count+= 1
    num_sensitive_words = count
    
    return num_sensitive_words


def get_path_depths(path):
    '''
    '''
    
    # get path_length
    path_length = len(path)
    # get path_level
    path_level = len(path.split('/'))
    
    return path_length, path_level


def get_lengths(url, query):
    '''
    '''
    
    # url_char_length
    url_char_length = len(url)
    # query_length
    query_length = len(query)
    
    return url_char_length, query_length


#################### Predict Input URL ####################


def predict_url_input():
    '''
    '''
    
    # parse URL
    url, hostname, path, query = url_parse()
    # get num_dash_url
    num_dash_url = get_num_dash_url(url)
    # get num_dot_url
    num_dot_url = get_num_dot_url(url)
    # get num_numerics
    num_numerics = get_num_numerics(url)
    # get num_queries
    num_queries = get_num_queries(query)
    # get num_sensitive_words
    num_sensitive_words = get_num_sensitive_words(hostname, path, query)
    # get path details
    path_length, path_level = get_path_depths(path)
    # get lengths
    url_char_length, query_length = get_lengths(url, query)
    # get fitted objects
    scaler, kmeans, model = fit_data()
    # create DataFrame for new observation
    new_obs_df = pd.DataFrame(
                    data={(num_dash_url, num_dot_url, num_numerics,
                           num_queries, num_sensitive_words,
                           path_length, path_level, query_length,
                           url_char_length, -10)},
                    columns=['num_dash_url', 'num_dot_url', 'num_numerics',
                             'num_queries', 'num_sensitive_words',
                             'path_length', 'path_level', 'query_length',
                             'url_char_length', 'dash_dot_clstr_2'])
    # scale data for kmeans and models
    new_obs_df = pd.DataFrame(scaler.transform(new_obs_df),
                              index=new_obs_df.index,
                              columns=new_obs_df.columns)
    # use fitted kmeans object to predict cluster
    if kmeans.predict(new_obs_df[['num_dash_url', 'num_dot_url']]) == 2:
        new_obs_df['dash_dot_clstr_2'] = 1
    else:
        new_obs_df['dash_dot_clstr_2'] = 0
    # obtain prediction and probability
    pred = model.predict(new_obs_df)
    prob = model.predict_proba(new_obs_df)
    # print statement of predicted legitimacy and likelihood
    st.text('')
    if url == '':
        pass
    else:
        if pred == 0:
            st.text(f'The URL provided is predicted {prob[0][0]:.0%} likely \
to be legitimate.')
            if prob[0][0] >= 0.75:
                st.text('It doesn\'t seem suspicious, but use your best \
judgment.')
            elif prob[0][0] >= 0.5:
                st.text('Feels pretty safe, but proceed with care.')
        else:
            st.text(f'The URL provided is {prob[0][1]:.0%} suspected of being \
illegitimate')
            if prob[0][1] >= 0.75:
                st.text('No good waits on the other side of this.')
            elif prob[0][1] >= 0.5:
                st.text('Best be cautious and not visit this site.')


#################### Begin Web Portal Interface ####################

# set header title and image
st.title('&nbsp;&nbsp;Is the URL a phishing attempt? Check! *')
st.image('https://raw.githubusercontent.com/ray-zapata/project_classification_phishing/main/assets/logo.jpg')

# begin function
predict_url_input()

# page breaks because I don't quite know a better way yet
components.html('')
components.html('')
components.html('')

# github repo link
link = '[GitHub](https://github.com/ray-zapata/project_classification_phishing)'
st.markdown(link, unsafe_allow_html=True)

#disclaimer
st.text('''
* For entertainment purposes only, does not claim to prevent
  or treat any computer illness or ensure personal security.
  ''')
