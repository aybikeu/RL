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

#basis = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_1R2.csv', sep=',')
basis = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS2.csv', sep=',')
basis.set_index('Unnamed: 0', inplace=True)
basis['period_nonlinear'] = basis['period'].apply(lambda x: np.exp(-0.2*x))

features = ['actual rem demand','satisfied demand','collected debris', 'resource']

n_features = len(features)

theta = np.ones(n_features)

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
resource = 5
phi_sa = {}

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

for _ in range(10000):


    G_disrupted = nx.Graph()
    G_disrupted.add_nodes_from(range(n_nodes))

    G2 = nx.Graph()
    G2.add_weighted_edges_from(EdgeListWeighted, weight='debris')

    state, actions, Schedule, reachable_nodes = sim.buildEnvironment(explored_states, state_dict, G, G2, G_disrupted, ActionList, supply_nodes)

    Qmatrix, action, basis = sim.sample(state, actions, supply_nodes, resource, Qmatrix, Schedule, QalphaMatrix, G_disrupted, G2, G, EdgeList, reachable_nodes,
           ActionList, dist, phi_sa, total_debris, total_supply, explored_states, state_dict, id_dict, id_counter)

    #basis_sa = basis.query('s== {} & a=={}'.format(state.ID, action))
    basis_sa = phi_sa[(state.ID, action)]
    bas = np.asarray(basis_sa)
    basis_sa = bas[[2,4,5,6]]  #Get the necessary features


    theta_next = theta - step_size*((Qmatrix.iloc[state.ID][action] - np.dot(basis_sa,theta))*basis_sa)

    theta = theta_next


#Extract the optimal policy: action for each state

valid_state_num = len(state_dict)/4
policy = funcs2.extractPolicy(Qmatrix, valid_state_num)