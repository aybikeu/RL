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
from sklearn import preprocessing


df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS3.csv', sep=',')
#df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS2.csv', sep=',')
#df = pd.read_csv('basis.csv', sep=',')
df.set_index('Unnamed: 0', inplace=True)

df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS3.csv',sep=',')
#df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS2.csv',sep=',')
df_q.set_index('Unnamed: 0', inplace=True)


q_column = pd.DataFrame(index=df.index.copy(), columns=['q_val'])
for i, row in df.iterrows():
    i_str = i[1:-1]
    st , act = i_str.split(',')
    q_column.loc[i]['q_val'] = df_q.iloc[int(st)][int(act)]


# df_merged = df.merge(q_column, left_index=True, right_index=True)
# df_merged.to_csv('FULLBASIS_1.csv', sep=',')

#features = ['degree', 'betw', 'resource','demand','collected debris', 'satisfied demand','period_nonlinear']
#features = ['degree', 'betw', 'resource','demand','collected debris', 'period_nonlinear']
# features = ['Action Betw','resource','satisfied demand', 'mean rem demand','actual rem demand','collected debris',
#             'satisfied demand','period', 'exponential period', 'degree']

#features = [0,1,2,5]
#features = ['actual rem demand','satisfied demand','collected debris', 'resource']
features = [3,7,10]

# plt.scatter(df['collected debris'], q_column['q_val'])
# plt.show()

x = df[features]
#ydf_merged = df['qval']
y = q_column['q_val']

df_xy = df.copy()
df_xy['Qval'] = y
df_xy.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/xNy_INS3.csv')


# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
# vif["features"] = x.columns

x = preprocessing.scale(x)
x = pd.DataFrame(x)

kf = KFold(n_splits=5, shuffle=True)

sum=0
model = LinearRegression()

results=[]
for train, test in kf.split(df):
    regr = model.fit(x.iloc[list(train)][:],y[list(train)])
    r = regr.score(x.iloc[list(test)][:],y[list(test)])
    results.append(r)

mean_score = np.array(results).mean()
print 'Results:' + str(results) + " Mean score:" + str(mean_score)

## Check the linear regression assumptions
# residuals = lr_predicted_y - data['y']
# plt.scatter(x=lr_predicted_y, y=residuals)
# plt.title('residuals')
# probplot(residuals, plot=plt)
##########################################

# X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
#
# train_data = pd.concat([X_train,y_train],axis=1)
#
# corrmat = train_data.corr()
# plt.figure(figsize=(5,5))
# g = sns.heatmap(corrmat, annot=True)
# plt.show()
# #
# #

# dt = pd.concat([x,y],axis=1)
# sns.pairplot(dt, size=2.5)
# plt.show()


# model.fit(X_train,y_train)
# y_predictions = model.predict(X_test)
#
# print 'R-squared:', model.score(X_test,y_test)


