import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import App_funcs as funcs
import warnings
warnings.filterwarnings('ignore')


q_column, df = funcs.load_data()

features = [0,1,2,3,4,5,6,7,8,9,10,11]

x = df[features]
y = q_column['q_val']


# plt.scatter(df['collected debris'], q_column['q_val'])
# data = pd.concat([x, y], axis=1)
# sns.pairplot(data, size=2.5)
# plt.show()
# plt.show()

#vif, corrmat = funcs.check_correlation(x,y)

#x = funcs.feature_scaling(x)
kf = KFold(n_splits=5, shuffle=True)

##### LINEAR REGRESSION
results, mse_test, mse_train, mse_all = funcs.lin_regression(kf, df, x, y)
mean_score = np.array(results).mean()
mean_mse = np.array(mse_test).mean()
print 'Results:' + str(results) + " Mean score:" + str(mean_score)
print 'MSE (test):' + str(mse_test) + " Mean score:" + str(mean_mse)
print 'MSE (train):' + str(mse_test) + " Mean score:" + str(np.array(mse_train).mean())
print 'MSE (all):' + str(mse_test) + " Mean score:" + str(mse_all)

#### RANDOM FOREST

mse, errors, accuracy, mape, feature_importances = funcs.random_forest(x, y)
# Print out the mean absolute error (mae)
print('MAE:', round(np.mean(errors), 2), 'degrees.')
print('Accuracy:', round(accuracy, 2), '%.')
print('MSE')
print(feature_importances)
