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
from scipy import stats
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_INS8.csv', sep=',')
df.set_index('Unnamed: 0', inplace=True)

df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS8.csv',sep=',')
df_q.set_index('Unnamed: 0', inplace=True)

#
# q_column = pd.DataFrame(index=df.index.copy(), columns=['q_val'])
# for i, row in df.iterrows():
#     i_str = i[1:-1]
#     st , act = i_str.split(',')
#     q_column.loc[i]['q_val'] = df_q.iloc[int(st)][int(act)]

q_column = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_column_INS8.csv')
q_column.index = df.index.copy()
q_column.drop(['Unnamed: 0'], axis=1, inplace=True)
q_column.columns = ['q_val']


features = [0,1,2,3,4,5,6,7,8,9,10,11]

# plt.scatter(df['collected debris'], q_column['q_val'])
# plt.show()

x = df[features]
#ydf_merged = df['qval']
y = q_column['q_val']

df_xy = df.copy()
df_xy['Qval'] = y
df_xy.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/xNy_INS2_V3.csv')

# features = [3,4]
x = df[features]
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns

# sc = StandardScaler()
# x = sc.fit_transform(x)


kf = KFold(n_splits=5, shuffle=True)

sum=0
model = LinearRegression()

results=[]
i=0
mse = []
for train, test in kf.split(df):

    colors = ['c', 'g', 'r', 'pink', 'teal' ]
    regr = model.fit(x.iloc[list(train)][:],y[list(train)])
    r = regr.score(x.iloc[list(test)][:],y[list(test)])
    m = (((regr.predict(x.iloc[list(test)][:]) - y[list(test)])**2).sum())/len(test)
    results.append(r)
    mse.append(m)

    # lr_predicted_y = regr.predict(x.iloc[list(test)][:])
    # residuals1 = lr_predicted_y - y[list(test)]
    # residuals2 = regr.predict(x.iloc[list(train)][:]) - y[list(train)]
    # sse = ((regr.predict(x) - y)**2).sum()
    # mse = sse/df.shape[0]
    # sst = ((regr.predict(x) - y.mean())**2).sum()
    # r_sq = 1 - (sse/sst)
    # plt.scatter(x=lr_predicted_y, y=residuals1, c = colors[i])
    # plt.scatter(x=regr.predict(x.iloc[list(train)][:]), y=residuals2, c='orange')
    #i += 1

    # plt.title('residuals')
    # plt.ylabel('residuals')
    # plt.xlabel('Predicted y')
    # plt.show()

    #stats.probplot(residuals, plot=plt)
#
# plt.title('residuals')
# plt.ylabel('residuals')
# plt.xlabel('Predicted y')
# plt.show()

mean_score = np.array(results).mean()
mean_mse = np.array(mse).mean()
print 'Results:' + str(results) + " Mean score:" + str(mean_score)
print 'MSE:' + str(mse) + " Mean score:" + str(mean_mse)
#
# scores = cross_val_score(model, x, y, cv=5)
# print 'Scores:' + str(scores) + " Mean score:" + str(scores.mean())

### Try Random Forest
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators = 2)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
score = rf.score(X_test, y_test)
print('R^2 score:', score)

errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error for Random Forest:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(feature_importances)

#Plot the feature importances
sort_ind = np.argsort(importances) #ascending order
sorted_importances = np.asarray(importances)[sort_ind]
x_values = list(range(len(importances)))
plt.bar(x_values, sorted_importances, orientation = 'vertical')
plt.xticks(x_values, x.columns[sort_ind], rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.tight_layout()
plt.show()
################# RANDOM FOREST END
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

train_data = pd.concat([X_train,y_train],axis=1)

corrmat = train_data.corr()
plt.figure(figsize=(5,5))
g = sns.heatmap(corrmat, annot=True)
plt.show()
# #
# #

# dt = pd.concat([x,y],axis=1)
# sns.pairplot(dt, size=2.5)
# plt.show()


# model.fit(X_train,y_train)
# y_predictions = model.predict(X_test)
#
# print 'R-squared:', model.score(X_test,y_test)


