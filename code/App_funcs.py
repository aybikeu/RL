import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

def load_data():
    df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_INS8.csv', sep=',')
    df.set_index('Unnamed: 0', inplace=True)

    df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS8.csv', sep=',')
    df_q.set_index('Unnamed: 0', inplace=True)

    # df['index_col'] = df.index
    # df['index_col'] = df['index_col'].apply(lambda x: x[1:-1].split(','))
    # df['state'] = df['index_col'].apply(lambda x: int(x[0]))
    # df['action'] = df['index_col'].apply(lambda x: int(x[1]))
    #
    # q_dict = df_q.to_dict()
    # q_column = dict()
    # for i in range(df.shape[0]):
    #     state = df.iloc[i]['state']
    #     action = df.iloc[i]['action']
    #     #q_column.iloc[i]['q_val'] = df_q.iloc[state][action]
    #     q_column[i] = q_dict[str(action)][state]
    #     print(i)
    #
    # qq = pd.DataFrame.from_dict(q_column.values())
    # qq.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_column_INS7.csv')

    q_column = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_column_INS8.csv')
    q_column.index = df.index.copy()
    q_column.drop(['Unnamed: 0'], axis=1, inplace=True)
    q_column.columns = ['q_val']

    return q_column, df

def lin_regression(kf, df, x, y):

    model = LinearRegression()
    results = []
    mse = []; mse_train=[]

    for train, test in kf.split(df):

        regr = model.fit(x.iloc[list(train)][:], y[list(train)])
        r = regr.score(x.iloc[list(test)][:], y[list(test)])
        m_test = (((regr.predict(x.iloc[list(test)][:]) - y[list(test)]) ** 2).sum()) / (2*len(test))
        m_train = (((regr.predict(x.iloc[list(train)][:]) - y[list(train)]) ** 2).sum()) / (2 * len(train))
        results.append(r)
        mse.append(m_test)
        mse_train.append(m_train)

        #residual_check(regr, x, y, test, train)

    model2 = LinearRegression()
    regr = model.fit(x, y)
    mse_all = (((regr.predict(x) - y) ** 2).sum()) / (2 * x.shape[0])
    error = regr.predict(x) - y
    gradient = np.linalg.norm(np.dot(x.T,error)/x.shape[0])
    print('Final gradient:', gradient)

    return results, mse, mse_train, mse_all

def residual_check(regr, x, y, test, train ):

    #colors = ['c', 'g', 'r', 'pink', 'teal']
    residuals_test = regr.predict(x.iloc[list(test)][:]) - y[list(test)]
    residuals_train = regr.predict(x.iloc[list(train)][:]) - y[list(train)]

    plt.scatter(x=regr.predict(x.iloc[list(test)][:]), y=residuals_test, c = 'teal')
    plt.scatter(x=regr.predict(x.iloc[list(train)][:]), y=residuals_train, c='orange')

    plt.title('residuals')
    plt.ylabel('residuals')
    plt.xlabel('Predicted y')
    plt.show()

    #check normality
    #stats.probplot(residuals, plot=plt)

def random_forest(x, y):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    score = rf.score(X_test, y_test)
    print('R^2 score:', score)

    errors = abs(predictions - y_test)
    mape = 100 * (errors / y_test)
    mse = np.square(errors)/ (2*len(y_test))
    accuracy = 100 - np.mean(mape)

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x.columns, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)


    plot_feat_importances(importances,x)

    return mse, errors, accuracy, mape, feature_importances

def plot_feat_importances(importances, x):

    sort_ind = np.argsort(importances)  # ascending order
    sorted_importances = np.asarray(importances)[sort_ind]
    sorted_importances = np.asarray([round(j, 3) for j in sorted_importances])
    x_values = list(range(len(importances)))

    fig = plt.figure()
    plt.bar(x_values, sorted_importances, orientation='vertical')
    for a, b in zip(x_values, sorted_importances):
        plt.text(a-0.4, b, str(b))
    plt.xticks(x_values, x.columns[sort_ind], rotation='vertical')
    plt.ylabel('Importance');
    plt.xlabel('Variable');
    plt.title('Variable Importances')
    plt.tight_layout()
    plt.savefig('C:/Users/ulusan.a/Desktop/RL_rep/RL/figures/FeatureImp_INS8_5.png')
    plt.show()

def check_correlation(x, y):

    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["features"] = x.columns

    data = pd.concat([x, y], axis=1)
    corrmat = data.corr()
    plt.figure(figsize=(5, 5))
    g = sns.heatmap(corrmat, annot=True)
    plt.tight_layout()
    plt.show()

    return vif, corrmat

def feature_scaling(x):

    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x

def lasso_regression(kf, df, x, y):

    model = linear_model.Lasso(alpha = 0.1)
    results = []
    mse = [];
    mse_train = []
    sum_coef = np.zeros(x.shape[1])

    for train, test in kf.split(df):
        regr = model.fit(x.iloc[list(train)][:], y[list(train)])
        r = regr.score(x.iloc[list(test)][:], y[list(test)])
        m_test = (((regr.predict(x.iloc[list(test)][:]) - y[list(test)]) ** 2).sum()) / (2 * len(test))
        m_train = (((regr.predict(x.iloc[list(train)][:]) - y[list(train)]) ** 2).sum()) / (2 * len(train))
        results.append(r)
        mse.append(m_test)
        mse_train.append(m_train)

        sum_coef += regr.coef_

        # residual_check(regr, x, y, test, train)

    # model2 = LinearRegression()
    # regr = model.fit(x, y)
    # mse_all = (((regr.predict(x) - y) ** 2).sum()) / (2 * x.shape[0])
    # error = regr.predict(x) - y
    # gradient = np.linalg.norm(np.dot(x.T, error) / x.shape[0])
    # print('Final gradient:', gradient)

    mean_coef = sum_coef / kf.n_splits
    return results, mse, mse_train, mean_coef

