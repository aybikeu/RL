import state2 as st
import pandas as pd
import numpy as np
import networkx as nx
import funcs2
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *
import math
import csv
import random
from copy import copy
import SNEBC
from nested_dict import nested_dict
import sampleSimulation as sim
from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


#df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS2_V3.csv', sep=',')
df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_INS8.csv', sep=',')
df.set_index('Unnamed: 0', inplace=True)

#df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS2_V3_2.csv',sep=',')
df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS8.csv',sep=',')
df_q.set_index('Unnamed: 0', inplace=True)

# df['index_col'] = df.index
# df['index_col'] = df['index_col'].apply(lambda x: x[1:-1].split(','))
# df['state'] = df['index_col'].apply(lambda x: int(x[0]))
# df['action'] = df['index_col'].apply(lambda x: int(x[1]))
#
# #q_column = pd.DataFrame(index=df.index.copy(), columns=['q_val'], data = np.zeros(df.shape[0]))
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
# qq.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_column_INS8.csv')

q_column = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_column_INS8.csv')

q_column.index = df.index.copy()
q_column.drop(['Unnamed: 0'], axis=1, inplace=True)
q_column.columns = ['q_val']

epochs = 10

n_features = 11
#Put the intercept as an extra feature
theta = np.ones(n_features + 1)
target = q_column
step_size = 0.001
sc = StandardScaler()
df = sc.fit_transform(df)
#df = preprocessing.scale(df, axis=1)

def gradient_descent2(df, target, step_size, iterations):

    theta = np.random.randn(df.shape[1])
    m = df.shape[0]
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,df.shape[1]))
    X = df
    y = target['q_val'].values
    #for it in range(iterations):
    norm_gradient = 1
    it = 0
    while norm_gradient >= 0.0001:
        prediction = np.dot(X,theta)
        error = prediction - y
        gradient = (np.dot(X.T,error)/m)
        theta_next = theta - step_size* gradient
        theta = theta_next
        theta_history[it,:] = theta.T
        cost_history[it] = (np.square(error).sum())/(2*m)

        norm_gradient = norm(gradient)
        print('Iteration {}: MSE: {} - Norm Gradient: {}'.format(it,cost_history[it], norm_gradient))

        #step_size -= 1/(iterations+1)
        it += 1

    return theta_history, cost_history, norm_gradient, it

def sgd(df, iterations, target, step_size):

    theta = np.random.randn(df.shape[1])
    m = df.shape[0]
    cost_history_ex00 = np.zeros(iterations)
    cost_history = np.zeros(iterations)
    gradient_history = np.zeros(iterations)
    target = target['q_val'].values
    for ep in range(iterations):

        it = 0
        # sum_mse_00 = 0
        # examples = range(m)
        # cost_history_it = np.zeros(m)
        #while examples:
        examples = random.sample(range(m),m)
        while it != len(examples):

            #ex = random.randint(0, len(examples)-1) #Pick each example randomly
            #examples.pop(ex)
            #bas = df.iloc[ex][:]
            ex = examples[it]
            bas = df[ex][:]

            prediction = np.dot(bas, theta)
            error = prediction - target[ex]
            gradient = bas * error
            theta_next = theta - step_size * gradient
            theta = theta_next

            #sum_mse_00 += (np.square(sum(df[0][:]* theta) - target.iloc[0]['q_val']).sum()) / (2 * 1)

            #total_error = sum((target['q_val'].values - np.dot(df, theta))**2)/(2*m)
            #total_gradient_norm = norm(np.dot(df.T,total_error)/m)
            #print('Iteration {} - Example {}: MSE : {} - Norm Gradient: {}'.format(ep, it, total_error, 'NAN'))
            #print('Iteration {} - Example {} '.format(ep,it))
            #cost_history_it[it] = total_error
            it += 1

        cost_history[ep] = sum((target - np.dot(df, theta))**2)/(2*m)
        #cost_history_ex00[ep] = sum_mse_00/m
        gradient_history[ep] = norm(np.dot(df.T,cost_history[ep])/m)
        print('Iteration {} - MSE {} - Gradient {}'.format(ep, cost_history[ep], gradient_history[ep]))
        #step_size -= 1/(iterations+1)

    return cost_history, cost_history_ex00, gradient_history, ep

# theta_history, cost_history, last_gradient, max_it = gradient_descent2(df, target, step_size,epochs)
# plot(cost_history[0:max_it])

cost_history, cost_history_ex00, last_gradient, max_it  = sgd(df, epochs, target, step_size)