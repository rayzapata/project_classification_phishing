#Z0096


# import from python libraries and modules
import pandas as pd
import numpy as np

# import visualization tools
import matplotlib.pyplot as plt

# import modeling tools
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, explained_variance_score


#################### Create & Test Models ####################


def train_model(X, y, model, model_name):
    '''

    Takes in the X_train and y_train, model object and model name, fit the
    model and returns predictions and a dictionary containg the model RMSE
    and R^2 scores on train

    '''

    # fit model to X_train_scaled
    model.fit(X, y)
    # predict X_train
    predictions = model.predict(X)
    # get rmse and r^2 for model predictions on X
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict


def model_testing(X, y, model, model_name):
    '''

    Takes in the X and y for validate or test, model object and model name and
    returns predictions and a dictionary containg the model RMSE and R^2 scores
    on validate or test

    '''
    
    # obtain predictions on X
    predictions = model.predict(X)
    # get for performance and assign them to dictionary
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict


#################### Model Performance ####################


def get_metrics(true, predicted, display=False):
    '''

    Takes in the true and predicted values and returns the rmse and r^2 for the
    model performance

    '''
    
    rmse = mean_squared_error(true, predicted, squared=False)
    r2 = explained_variance_score(true, predicted)
    if display == True:
        print(f'Model RMSE: {rmse:.2g}')
        print(f'       R^2: {r2:.2g}')
    return rmse, r2


def plot_residuals(y_true, y_predicted):
    '''

    Takes in 1 to 4 prediction sets and returns a configured scatterplot of the
    residual errors of those predictions against the passed true set
    
    '''

    # set figure dimensions
    plt.figure(figsize=(60, 40))
    plt.rcParams['legend.title_fontsize'] = 50
    # scatterplot for each up to four predictions passed in list
    plt.scatter(y_true, (y_predicted.iloc[0:,0] - y_true), alpha=1,
                    color='cyan', s=250, label=y_predicted.iloc[0:,0].name,
                    edgecolors='black')
    if len(y_predicted.columns) > 1:
        plt.scatter(y_true, (y_predicted.iloc[0:,1] - y_true), alpha=0.75, 
                    color='magenta', s=250, label=y_predicted.iloc[0:,1].name, 
                    edgecolors='black')
    if len(y_predicted.columns) > 2:
        plt.scatter(y_true, (y_predicted.iloc[0:,2] - y_true), alpha=0.75, 
                    color='yellow', s=250, label=y_predicted.iloc[0:,2].name, 
                    edgecolors='black')
    if len(y_predicted.columns) > 3:
        plt.scatter(y_true, (y_predicted.iloc[0:,3] - y_true), alpha=0.5, 
                    color='black', s=250, label=y_predicted.iloc[0:,3].name, 
                    edgecolors='white')
    if len(y_predicted.columns) > 4:
        return 'Can only plot up to four models\' predictions'
    # add zero line for ease of readability
    plt.axhline(label='', color='red', linewidth=5, linestyle='dashed',
                    alpha=0.25)
    # model legend
    plt.legend(title='Models', loc=(0.025,0.05), fontsize=50)
    
    # set labels and title
    plt.xlabel('\nTrue Value\n', fontsize=50)
    plt.xticks(fontsize=50)
    plt.ylabel('\nPredicted Value Error\n', fontsize=50)
    plt.yticks(fontsize=50)
    plt.title(f'\nPrediction Residuals of {y_true.name}\n', fontsize=50)
    # show plot
    plt.show()
