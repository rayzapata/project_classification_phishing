#Z0096


# import from python libraries and modules
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# import visualization tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import feature exploration tools
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# import statistical tests
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.metrics import confusion_matrix, classification_report


#################### Explore Data ######################


def elbow_plot(df, col_list):
    '''

    Takes in a DataFrame and column list to use below method to find
    changes in inertia for increasing k in cluster creation methodology

    '''

    # set figure parameters
    plt.figure(figsize=(30, 15))
    # create series and apply increasing k values to test for inertia
    pd.Series({k: KMeans(k).fit(df[col_list])\
                            .inertia_ for k in range(2, 15)}).plot(marker='*')
    # define plot labels and visual components
    plt.xticks(range(2, 15))
    plt.xlabel('$k$')
    plt.ylabel('Inertia')
    plt.ylim(0,50000)
    plt.title('Changes in Inertia for Increasing $k$')
    plt.show()


def explore_clusters(df, col_list, k=2):
    '''

    Takes in a DataFrame, column list, and optional integer value for
    k to create clusters for the purpose of exploration, returns a
    DataFrame containing cluster group numbers and cluster centers

    '''

    # create kmeans object
    kmeans = KMeans(n_clusters=k, random_state=19)
    # fit kmeans
    kmeans.fit(df[col_list])
    # store predictions
    cluster_df = pd.DataFrame(kmeans.predict(df[col_list]), index=df.index,
                                                        columns=['cluster'])
    cluster_df = pd.concat((df[col_list], cluster_df), axis=1)
    # store centers
    center_df = cluster_df.groupby('cluster')[col_list].mean()
    
    return cluster_df, center_df, kmeans


#################### Visualize Data ####################


def plot_heat(df, method='pearson'):
    '''

    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all chosen variables


    '''

    n_vars = len(df.columns.to_list())
    # Set up large figure size for easy legibility
    plt.figure(figsize=(n_vars + 5, n_vars + 1))
    # assign pd.corr() output to variable and create a mask to remove
    # redundancy from graphic
    corr = df.corr(method=method)
    mask = np.triu(corr, k=0)
    # define custom cmap for heatmap where the darker the reds the more
    # positive and vice versa for blues
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # create graphic with zero centered cmap and annotations set to one
    # significant figure
    sns.heatmap(corr, cmap=cmap, center=0, annot=True, fmt=".1g", square=True,
                mask=mask, cbar_kws={
                                     'shrink':0.5,
                                     'aspect':50,
                                     'use_gridspec':False,
                                     'anchor':(-0.75,0.75)
                                      })
    # format xticks for improved legibility and clarity
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.title('Correlation Heatmap')
    plt.show()


def target_heat(df, target, method='pearson'):
    '''

    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all variables


    '''

    # define variable for corr matrix
    heat_churn = df.corr()[target][:-1]
    # set figure size
    fig, ax = plt.subplots(figsize=(30, 1))
    # define cmap for chosen color palette
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # plot matrix turned to DataFrame
    sns.heatmap(heat_churn.to_frame().T, cmap=cmap, center=0,
                annot=True, fmt=".1g", cbar=False, square=True)
    #  improve readability of xticks, remove churn ytick
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.yticks(ticks=[])
    # set title and print graphic
    plt.title(f'Correlation to {target}\n')
    plt.show()


def plot_univariate(data, variable):
    '''

    This function takes the passed DataFrame the requested and plots a
    configured boxenplot and distrubtion for it side-by-side

    '''

    # set figure dimensions
    plt.figure(figsize=(30,8))
    # start subplot 1 for boxenplot
    plt.subplot(1, 2, 1)
    sns.boxenplot(x=variable, data=data)
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.title('Enchanced Box Plot', fontsize=25)
    # start subplot 2 for displot
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=variable, element='step', kde=True, color='cyan',
                                line_kws={'linestyle':'dashdot', 'alpha':1})
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Distribution', fontsize=20)
    # set layout and show plot
    plt.suptitle(f'{variable} $[n = {data[variable].count():,}]$', fontsize=25)
    plt.tight_layout()
    plt.show()


def plot_discrete_to_continous(data, discrete_var, continous_var, hue=None,
                            swarm_n=2000, r_type='pearson', random_state=19):
    '''

    Takes in a DataFrame and lists of discrere and continuous variables and
    plots a boxenplot, swarmplot, and regplot for each against the other,
    providing either the pearson (default) or spearman r measurement in the
    title

    '''

    # choose coefficient
    if r_type == 'pearson':
        r = pearsonr(data[discrete_var], data[continous_var])[0]
    elif r_type =='spearman':
        r = spearmanr(data[discrete_var], data[continous_var])[0]
    # set figure dimensions
    plt.figure(figsize=(30,10))
    # start subplot 1 for boxplot
    plt.subplot(1, 3, 1)
    sns.boxenplot(x=discrete_var, y=continous_var, data=data)
    plt.xlabel('')
    plt.ylabel(f'{continous_var}', fontsize=20)
    # start subplot 2 for boxplot
    plt.subplot(1, 3, 2)
    sns.swarmplot(x=discrete_var, y=continous_var, data=data.sample(n=swarm_n,
                                                    random_state=random_state))
    plt.xlabel(f'{discrete_var}', fontsize=20)
    plt.ylabel('')
    # start subplot 3 for boxplot
    plt.subplot(1, 3, 3)
    sns.regplot(x=discrete_var, y=continous_var, data=data, marker='*',
                                                    line_kws={'color':'red'})
    plt.xlabel('')
    plt.ylabel('')
    # set title for graphic and output
    plt.suptitle(f'{discrete_var} to {continous_var} $[r = {r:.2f}]$',
                                                                fontsize=25)
    plt.tight_layout()
    plt.show()


def plot_joint(data, x, y, r_type='pearson'):
    '''

    Takes in a DataFrame and the specified x, y variables and plots a
    configured joint plot with the pearson (default) or spearman r measurement
    in the title

    '''

    # choose coefficient
    if r_type == 'pearson':
        r = pearsonr(data[x], data[y])[0]
    elif r_type =='spearman':
        r = spearmanr(data[x], data[y])[0]
    # plot jointplot of continuous variables
    sns.jointplot(x, y, data, kind='reg', height=10,
                  joint_kws={'marker':'+', 'line_kws':{'color':'red'}},
                  marginal_kws={'color':'cyan'})
    # set labels for x, y axes
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    # set title of compared variables
    plt.suptitle(f'{x} to {y} $[r = {r:.2f}]$')
    plt.tight_layout()
    # show plot
    plt.show()


def corr_test(data, x, y, alpha=0.05, r_type='pearson'):
    '''

    Performs a pearson or spearman correlation test and returns the r
    measurement as well as comparing the return p valued to the pass or
    default significance level, outputs whether to reject or fail to
    reject the null hypothesis
    
    '''
    
    # obtain r, p values
    if r_type == 'pearson':
        r, p = pearsonr(data[x], data[y])
    if r_type == 'spearman':
        r, p = spearmanr(data[x], data[y])
    # print reject/fail statement
    print(f'''{r_type:>10} r = {r:.2g}
+--------------------+''')
    if p < alpha:
        print(f'''
        Due to p-value {p:.2g} being less than our significance level of \
{alpha}, we may reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')
    else:
        print(f'''
        Due to p-value {p:.2g} being greater than our significance level of \
{alpha}, we fail to reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')


def plot_clusters(cluster_df, center_df, x_var, y_var):
    '''

    Takes in cluster and centers DataFrame created by explore_clusters
    function and plots the passed x and y variables that make up that
    cluster group with different colors

    '''

    # define cluster_ column for better seaborn interpretation
    cluster_df['cluster_'] = 'cluster_' + cluster_df.cluster.astype(str)
    # set scatterplot and dimensions
    plt.figure(figsize=(28, 14))
    sns.scatterplot(x=x_var, y=y_var, data=cluster_df, hue='cluster_', s=100)
    # plot cluster centers
    center_df.plot.scatter(x=x_var, y=y_var, ax=plt.gca(), s=300, c='k',
                                        edgecolor='w', marker='$\\bar{x}$')
    # set labels and legend, show
    plt.xlabel(f'\n{x_var}\n', fontsize=20)
    plt.ylabel(f'\n{y_var}\n', fontsize=20)
    plt.title('\nClusters and Their Centers\n', fontsize=30)
    plt.legend(bbox_to_anchor=(0.95,0.95), fontsize=20)

    plt.show()


def plot_three_d_clusters(cluster_df, center_df, x_var, y_var, z_var):
    '''

    Takes in cluster and centers DataFrame created by explore_clusters
    function and creates a three dimesnional plot of the passed x, y,
    and z variables that make up that cluster group with different
    colors

    '''

    # set figure and axes
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')    
    # set clusters for each cluster passed in arguments
    # set x, y, z for cluster 0
    x0 = cluster_df[cluster_df['cluster'] == 0][x_var]
    y0 = cluster_df[cluster_df['cluster'] == 0][y_var]
    z0 = cluster_df[cluster_df['cluster'] == 0][z_var]
    # set x, y, z for cluster 1
    x1 = cluster_df[cluster_df['cluster'] == 1][x_var]
    y1 = cluster_df[cluster_df['cluster'] == 1][y_var]
    z1 = cluster_df[cluster_df['cluster'] == 1][z_var]
    # set x, y, z for each additional cluster
    if len(center_df) > 2:
        x2 = cluster_df[cluster_df['cluster'] == 2][x_var]
        y2 = cluster_df[cluster_df['cluster'] == 2][y_var]
        z2 = cluster_df[cluster_df['cluster'] == 2][z_var]
    if len(center_df) > 3:
        x3 = cluster_df[cluster_df['cluster'] == 3][x_var]
        y3 = cluster_df[cluster_df['cluster'] == 3][y_var]
        z3 = cluster_df[cluster_df['cluster'] == 3][z_var]
    if len(center_df) > 4:
        x4 = cluster_df[cluster_df['cluster'] == 4][x_var]
        y4 = cluster_df[cluster_df['cluster'] == 4][y_var]
        z4 = cluster_df[cluster_df['cluster'] == 4][z_var]
    if len(center_df) > 5:
        x5 = cluster_df[cluster_df['cluster'] == 5][x_var]
        y5 = cluster_df[cluster_df['cluster'] == 5][y_var]
        z5 = cluster_df[cluster_df['cluster'] == 5][z_var]
        
    # set centers for each cluster passed in arguments
    # set centers for clusters 0, 1
    zero_center = center_df[center_df.index == 0]
    one_center = center_df[center_df.index == 1]
    # set centers for each additional clusters
    if len(center_df) > 2:
        two_center = center_df[center_df.index == 2]
    if len(center_df) > 3:
        three_center = center_df[center_df.index == 3]
    if len(center_df) > 4:
        four_center = center_df[center_df.index == 4]
    if len(center_df) > 5:
        five_center = center_df[center_df.index == 5]
    if len(center_df) > 6:
        six_center = center_df[center_df.index == 6]
        
    # plot clusters and their centers for each cluster passed in arguments
    # plot cluster 0 with center
    ax.scatter(x0, y0, z0, s=100, c='c', edgecolor='k', marker='o',
                                                    label='Cluster 0')
    ax.scatter(zero_center[x_var], zero_center[y_var], zero_center[z_var],
                                    s=300, c='c', marker='$\\bar{x}$')
    # plot cluster 1 with center
    ax.scatter(x1, y1, z1, s=100, c='y', edgecolor='k', marker='o',
                                                    label='Cluster 1')
    ax.scatter(one_center[x_var], one_center[y_var], one_center[z_var],
                                    s=300, c='y', marker='$\\bar{x}$')
    # plot each additional cluster passed in arguments
    if len(center_df) > 2:
        ax.scatter(x2, y2, z2, s=100, c='m', edgecolor='k', marker='o',
                                                    label='Cluster 2')
        ax.scatter(two_center[x_var], two_center[y_var], two_center[z_var],
                                    s=300, c='m', marker='$\\bar{x}$')
    if len(center_df) > 3:
        ax.scatter(x3, y3, z3, s=100, c='k', edgecolor='w', marker='o',
                                                    label='Cluster 3')
        ax.scatter(three_center[x_var],three_center[y_var],three_center[z_var],
                                    s=300, c='k', marker='$\\bar{x}$')
    if len(center_df) > 4:
        ax.scatter(x4, y4, z4, s=100, c='r', edgecolor='k', marker='o',
                                                    label='Cluster 4')
        ax.scatter(four_center[x_var], four_center[y_var], four_center[z_var],
                                    s=300, c='r', marker='$\\bar{x}$')
    if len(center_df) > 5:
        ax.scatter(x5, y5, z5, s=100, c='g', edgecolor='k', marker='o',
                                                    label='Cluster 5')
        ax.scatter(five_center[x_var], five_center[y_var], five_center[z_var],
                                    s=300, c='g', marker='$\\bar{x}$')
    if len(center_df) > 6:
        ax.scatter(x6, y6, z6, s=100, c='b', edgecolor='k', marker='o',
                                                    label='Cluster 6')
        ax.scatter(six_center[x_var], six_center[y_var], six_center[z_var],
                                    s=300, c='b', marker='$\\bar{x}$')
        
    # set labels, title, and legend
    ax.set_xlabel(f'\n$x =$ {x_var}', fontsize=15)
    ax.set_ylabel(f'\n$y =$ {y_var}', fontsize=15)
    ax.set_zlabel(f'\n$z =$ {z_var}', fontsize=15)
    plt.title('Clusters and Their Centers', fontsize=30)
    plt.legend(bbox_to_anchor=(0.975,0.975), fontsize=15)

    plt.show()


#################### Feature Exploration ####################


def create_clusters(train, validate, test, col_list, cluster_name, k=2):
    '''

    Takes in the train, validate, and test DataFrames to create k
    clusters of passed list variables, fits the kmeans object to the
    training data, and transforms all three DataFrames before returning

    Requires passed cluster_name as a string to name created columns
    
    '''

    # scale data for distance in kmeans
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[col_list])
    validate_scaled = scaler.transform(validate[col_list])
    test_scaled = scaler.transform(test[col_list])
    # create kmeans object
    kmeans = KMeans(n_clusters=k, random_state=19)
    # fit kmeans
    kmeans.fit(train_scaled)
    # store predictions and encode on train
    train[f'{cluster_name}_clstr'] = kmeans.predict(train_scaled)
    train = pd.get_dummies(train, columns=[f'{cluster_name}_clstr'], drop_first=True)
    # store predictions and encode on validate
    validate[f'{cluster_name}_clstr'] = kmeans.predict(validate_scaled)
    validate = pd.get_dummies(validate, columns=[f'{cluster_name}_clstr'], drop_first=True)
    # store predictions and encode on test
    test[f'{cluster_name}_clstr'] = kmeans.predict(test_scaled)
    test = pd.get_dummies(test, columns=[f'{cluster_name}_clstr'], drop_first=True)
    
    return train, validate, test


def select_kbest(X, y, k=1, score_func=f_regression):
    '''

    Takes in the X, y train and an optional k values and score_func to use
    SelectKBest to return k (default=1) best variables for predicting the
    target of y
    
    '''

    # assign SelectKBest using f_regression and top two features default
    selector = SelectKBest(score_func=score_func, k=k)
    # fit selector to training set
    selector.fit(X, y)
    # assign and apply mask to DataFrame for column names
    mask = selector.get_support()
    top_k = X.columns[mask].to_list()
    return top_k


def select_rfe(X, y, n=1, model=LinearRegression(normalize=True), rank=False):
    '''

    Takes in the X, y train and an optional n values and model to use with
    RFE to return n (default=1) best variables for predicting the
    target of y, optionally can be used to output ranks of features in
    predictions

    '''

    # assign RFE using LinearRegression and top two features as default
    selector = RFE(estimator=model, n_features_to_select=n)
    # fit selector to training set
    selector.fit(X, y)
    # assign and apply mask to DataFrame for column names
    mask = selector.get_support()
    top_n = X.columns[mask].to_list()
    # check if rank=True
    if rank == True:
        # print DataFrame of rankings
        print(pd.DataFrame(X.columns, selector.ranking_,
                           [f'n={n} RFE Rankings']).sort_index())
    return top_n


#################### Measure Performance ####################


def two_sample_ttest(a, b, alpha=0.05, equal_var=True,
                        alternative='two-sided'):
    '''

    Perform T-Test using scipy.stats.ttest, prints whether to reject or accept
    the null hypothesis, as well as alpha, p-value, and t-value

    '''

    t, p = ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    null_hyp = f'there is no difference in {a.name} between the two populations'
    # print alpha and p-value
    print(f'''
  alpha: {alpha}
p-value: {p:.1g}''')
    # print if our p-value is less than our significance level
    if p < alpha:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we must reject the null hypothesis
        that {null_hyp}.
        ''')
    # print if our p-value is greater than our significance level
    else:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we fail to reject the null hypothesis
        that {null_hyp}.    
        ''')


def chi_test(cat, target, alpha=0.05):
    '''

    Takes in a category for comparison with the passed target and
    default alpha=0.05, then creates a crosstab DataFrame containing
    the observed values

    Performs a chi2_contingency to find p-value and ouputs a DataFrame
    of expected values, then prints alpha, p-value, whether to accept
    or reject null hypothesis, and the dataframes of observed and
    expected values

    More robust version of chi_test_lite

    '''

    # set observed DataFrame with crosstab
    observed = pd.crosstab(cat, target)
    a = observed.iloc[0,0]
    b = observed.iloc[0,1]
    c = observed.iloc[1,0]
    d = observed.iloc[1,1]
    # assign returned values from chi2_contigency
    chi2, p, degf, expected = chi2_contingency(observed)
    # set expected DataFrame from returned array
    expected = pd.DataFrame(expected)
    a2 = expected.iloc[0,0]
    b2 = expected.iloc[0,1]
    c2 = expected.iloc[1,0]
    d2 = expected.iloc[1,1]
    # set null hypothesis
    null_hyp = f'{target.name} is independent of {cat.name}'
    # print alpha and p-value
    print(f'''
  alpha: {alpha}
p-value: {p:.1g}''')
    # print if our p-value is less than our significance level
    if p < alpha:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we must reject the null hypothesis
        that {null_hyp}.''')
    # print if our p-value is greater than our significance level
    else:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we fail to reject the null hypothesis
        that {null_hyp}.''')
    # print observed and expected DataFrames side by side
    print(f'''
                       ** Observed **                        |       ** Expected **
                       --------------------------------------|--------------------------------------
                                     No Churn    Churn       |                     No Churn    Churn
                                                             |       
                       No Fiber      {a:<10.0f}  {b:<10.0f}  |       No Fiber      {a2:<10.0f}  {b2:<10.0f}
                                                             |       
                          Fiber      {c:<10.0f}  {d:<10.0f}  |          Fiber      {c2:<10.0f}  {d2:<10.0f}
    ''')


def chi_test_lite(cat, target, alpha=0.05):
    '''

    Takes in a category for comparison with the passed target and
    default alpha=0.05, then creates a crosstab DataFrame containing
    the observed values

    Performs a chi2_contingency to find p-value and ouputs a DataFrame
    of expected values, then prints alpha, p-value, and whether to
    accept or reject null hypothesis

    Simpler output version of chi_test
    
    '''

    # set observed DataFrame with crosstab
    observed = pd.crosstab(cat, target)
    a = observed.iloc[0,0]
    b = observed.iloc[0,1]
    c = observed.iloc[1,0]
    d = observed.iloc[1,1]
    # assign returned values from chi2_contigency
    chi2, p, degf, expected = chi2_contingency(observed)
    # set expected DataFrame from returned array
    expected = pd.DataFrame(expected)
    a2 = expected.iloc[0,0]
    b2 = expected.iloc[0,1]
    c2 = expected.iloc[1,0]
    d2 = expected.iloc[1,1]
    # set null hypothesis
    null_hyp = f'{target.name} is independent of {cat.name}'
    # print alpha and p-value
    print(f'''
  alpha: {alpha}
p-value: {p:.1g}''')
    # print if our p-value is less than our significance level
    if p < alpha:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we must reject the null hypothesis
        that {null_hyp}.
        ''')
    # print if our p-value is greater than our significance level
    else:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we fail to reject the null hypothesis
        that {null_hyp}.
        ''')


def cmatrix(y_true, y_pred):
    '''

    Takes in true and predicted values to create a confusion matrix,
    then ouputs dictionary holding the true pos, true, neg, false pos,
    and false neg rates discerned from the matrix

    Used in conjunction with model_report

    '''

    # assign TN, FN, TP, FP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #do math to find rates
    tpr = (tp / (tp + fn))
    tnr = (tn / (tn + fp))
    fpr = 1 - tnr
    fnr = 1 - tpr
    cmatrix_dict = {'tpr':tpr, 'tnr':tnr, 'fpr':fpr, 'fnr':fnr}

    return cmatrix_dict


def model_report(y_true, y_pred, lite=False):
    '''

    Takes in true and predicted values to create classificant report
    dictionary and uses cmatrix function to obtain positive and
    negative prediction rates, prints out table containing all metrics
    for the positive class of target

    '''

    # create dictionary for classification report and confusion matrix
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    cmatrix_dict = cmatrix(y_true, y_pred)
    # print formatted table with desired information for model report
    if lite == False:
        print(f'''
 _____________________________________________
|            *** Model  Report ***            |
|---------------------------------------------|
|                 Accuracy: {report_dict['accuracy']:>8.2%}          |
|       True Positive Rate: {cmatrix_dict['tpr']:>8.2%}          |
|      False Positive Rate: {cmatrix_dict['fpr']:>8.2%}          |
|       True Negative Rate: {cmatrix_dict['tnr']:>8.2%}          |
|      False Negative Rate: {cmatrix_dict['fnr']:>8.2%}          |
|                Precision: {report_dict['1']['precision']:>8.2%}          |
|                   Recall: {report_dict['1']['recall']:>8.2%}          |
|                 F1-Score: {report_dict['1']['f1-score']:>8.2%}          |
|                                             |
|         Positive Support: {report_dict['1']['support']:>8}          |
|         Negative Support: {report_dict['0']['support']:>8}          |
|            Total Support: {report_dict['macro avg']['support']:>8}          |
|_____________________________________________|''')
    elif lite == True:
         print(f'''             
 _____________________________________________
|            *** Model  Report ***            |
|---------------------------------------------|
|                 Accuracy: {report_dict['accuracy']:>8.2%}          |
|                Precision: {report_dict['1']['precision']:>8.2%}          |
|                   Recall: {report_dict['1']['recall']:>8.2%}          |
|            Total Support: {report_dict['macro avg']['support']:>8}          |
|_____________________________________________|''')


def validate(X, y, model, lite=False):
    '''

    Takes in feature DataFrame, true target, and fitted model to obtain
    model_report for model predictions on validation dataset

    Same function as final_test

    '''

    # assign model predictions on validate data
    y_pred = model.predict(X)
    model_report(y, y_pred, lite=lite)

    return y_pred


def final_test(X, y, model):
    '''

    Takes in feature DataFrame, true target, and fitted model to obtain
    model_report for model predictions on test dataset

    Same function as final_test

    '''

    # assign model predictions on test data
    y_pred = model.predict(X)
    # print model metrics on test data
    model_report(y, y_pred, lite=False)

    return y_pred
