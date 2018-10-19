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
import sampleSimulation as sim


n_features = 5
#Put the intercept as an extra feature
theta = np.ones(n_features + 1)

EdgeList = [(0,1), (0,2), (1,2), (1,3),
            (2,3), (2,4), (3,4)]

# EdgeListWeighted = [(0,1,1), (0,2,3), (1,2,3), (1,3,2),
#             (2,3,1), (2,4,3), (3,4,1)]

EdgeListWeighted = [(0,1,0.5), (0,2,1.5), (1,2,4), (1,3,0.5),
            (2,3,1), (2,4,3), (3,4,1)]

ActionList = dict(zip(EdgeList, range(len(EdgeList))))

G = nx.Graph()
G.add_weighted_edges_from(EdgeListWeighted,weight='debris')


initial_debris = [0.5,1.5,4,0.5,1,3,1] # for the second instance
#initial_debris = [1,3,3,2,1,3,1] #for all edges
initial_supply = [5,0,0,0,5] # for all nodes
demand_indicator= [0,0,1,1,0] # for all nodes
n_edges = len(initial_debris)
n_nodes = len(initial_supply)
n_actions = n_edges

max_demand = 5
xk = np.arange(1, max_demand+1)
pk = [0.1 , 0.2 , 0.4, 0.2, 0.1]
dist = stats.rv_discrete(name='dist', values=(xk, pk))

mean_dist = dist.mean()

# Figure out how many different combinations for states
n_debriscomb = 2**n_edges
n_supplycomb = reduce((lambda x,y: x*y), [j+1 for j in initial_supply])
n_demandcomb = reduce((lambda x,y: x*y), [j+1 for j in map(lambda x: x*max_demand, demand_indicator)])
#Integer demand values - simply rounded DOWN
initial_demand = map(lambda x: (x*mean_dist).round(0), demand_indicator) #For now equate the rem_demand of demand nodes to their exp value
supply_nodes = [i for i,p in enumerate(initial_supply) if p>0]

total_debris = sum(initial_debris)
total_supply = sum(initial_supply)

id_dict = {}
id_counter = 0

n_states = n_debriscomb * n_supplycomb * n_demandcomb

Qmatrix = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)

QalphaMatrix = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)

state_dict = {}
explored_states = []
resource = 1
phi_sa = {}
betw_centrality_service = {}
betw_centrality_regular = {}
betw_centrality_debris = {}
betw_centrality_regular_sp = {}

state_dict[(0, 'demand')] = initial_demand
state_dict[(0, 'debris')] = initial_debris
state_dict[(0, 'supply')] = initial_supply
state_dict[(0, 'period')] = 1
state_dict[(0, 'resource')] = 0



initial_state = st.State(initial_debris, initial_supply, initial_demand, 0, None)
id_counter, id_dict = initial_state.getStateIndex(id_counter, id_dict)
actions = funcs2.initializeActionSpace(supply_nodes, G, ActionList)
explored_states.append(initial_state.ID)

step_size = 0.1

# Get the actual optimal calculated to see if GD is working
df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS2.csv', sep=',')
df_q.set_index('Unnamed: 0', inplace=True)

df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS2.csv', sep=',')
df.set_index('Unnamed: 0', inplace=True)

q_column = pd.DataFrame(index=df.index.copy(), columns=['q_val'])
for i, row in df.iterrows():
    i_str = i[1:-1]
    st , act = i_str.split(',')
    q_column.loc[i]['q_val'] = df_q.iloc[int(st)][int(act)]

q_column.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_columns_INS2.csv')

for iteration_no in range(10000):


    G_restored = nx.Graph()
    G_restored.add_nodes_from(range(n_nodes))

    G2 = nx.Graph()
    G2.add_weighted_edges_from(EdgeListWeighted, weight='debris')

    state, actions, Schedule, reachable_nodes = sim.buildEnvironment(explored_states, state_dict, G, G2, G_restored, ActionList, supply_nodes)

    phi_sa, action, id_counter, new_state, reward, period, actions, betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp  = sim.sample(state, actions, supply_nodes, resource, Qmatrix, Schedule, QalphaMatrix, G_restored, G2, G, EdgeList, reachable_nodes,
                                                                               ActionList, dist, phi_sa, total_debris, total_supply, explored_states,
                                                                               state_dict, id_dict, id_counter, betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp )

    phi_sa, action_order, Basis, betw_centrality_service,  betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp = sim.new_state_basis(new_state,phi_sa, ActionList, state.cum_resource, G_restored, EdgeList, G2, total_debris, actions,
                                                                       betw_centrality_service,total_supply, betw_centrality_regular, betw_centrality_debris,betw_centrality_regular_sp)

    Q_pred_new_state = np.dot(Basis,theta)
    max_q = max(Q_pred_new_state)
    # m_i = Q_pred_new_state.index(max_q)
    # action_max = action_order[m_i]

    Qmatrix_previous = Qmatrix.copy() #Store Qmatrix's previous values

    Qmatrix.iloc[state.ID][action] = reward + max_q

    #basis_sa = basis.query('s== {} & a=={}'.format(state.ID, action))
    bas = np.asarray(phi_sa[(state.ID, action)])
    #basis_sa = bas[[2,4,5,6]]  #Get the necessary features

    #Qmatrix = funcs2.updatePredQ(Qmatrix, action, state, new_state, reward, theta, phi_sa)

    #target = Qmatrix.iloc[state.ID][action]
    #target = q_column.loc[str((state.ID, action))]['q_val'] #this is the optimal Q value calc from value iteration
    target = funcs2.Qtarget(state.ID,id_dict,action,q_column)
    error = target - np.dot(bas, theta)
    theta_next = theta + (0.001 * (error * bas))

    print('Iteration {} - Error:'.format(iteration_no), error)
    theta = theta_next


#Extract the optimal policy: action for each state

valid_state_num = len(state_dict)/5
policy = funcs2.extractPolicy(Qmatrix, valid_state_num)