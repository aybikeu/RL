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
import warnings
warnings.filterwarnings("ignore")


n_features = 11
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
df_q = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS2_V3_2.csv', sep=',')
df_q.set_index('Unnamed: 0', inplace=True)

df = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS2_V3.csv', sep=',')
df.set_index('Unnamed: 0', inplace=True)

q_column = pd.DataFrame(index=df.index.copy(), columns=['q_optimal', 'q_pred'])
for i, row in df.iterrows():
    i_str = i[1:-1]
    st , act = i_str.split(',')
    q_column.loc[i]['q_optimal'] = df_q.iloc[int(st)][int(act)]

q_column['q_pred']=0.0

q_column.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/q_columns_INS2_V3.csv')

#Q_predicted = pd.DataFrame(data = 0, index=df.index.copy(), columns=['q_predicted'])

n_training = shape(q_column)[0] #We know this is the leats # of training examples that should be found - if it finds more then its due to simulation, encountoring new examples
#n_training = 200
step_size = 0.0001
n_epochs = 50
s = []
# Create the hashing
state_mapper = funcs2.stateMapper()
example_no = 0
q_optimal={}
q_check = nested_dict()
for ep in range(n_epochs):
    train = {}
    example_no = 0
    iter = 0
    while not((example_no == n_training-1) or (iter > n_training *2)): #When you find that many training examples

        iter +=1
        G_restored = nx.Graph()
        G_restored.add_nodes_from(range(n_nodes))

        G2 = nx.Graph()
        G2.add_weighted_edges_from(EdgeListWeighted, weight='debris')

        state, actions, Schedule, reachable_nodes = sim.buildEnvironment(explored_states, state_dict, G, G2, G_restored, ActionList, supply_nodes)

        #state_copy = copy(state)

        phi_sa, action, id_counter, new_state, reward, period, actions, betw_centrality_service, \
        betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp, reachable_nodes  = sim.sample(state, actions, supply_nodes, resource, Qmatrix, Schedule, QalphaMatrix, G_restored, G2, G, EdgeList, reachable_nodes,
                                                                                   ActionList, dist, phi_sa, total_debris, total_supply, explored_states,
                                                                                   state_dict, id_dict, id_counter, betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp )



        phi_sa, action_order, Basis, betw_centrality_service, \
        betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp = sim.new_state_basis(new_state,phi_sa, ActionList, state.cum_resource, G_restored, EdgeList, G2, total_debris, actions,
                                                                           betw_centrality_service,total_supply, betw_centrality_regular, betw_centrality_debris,betw_centrality_regular_sp, reachable_nodes, resource)


        #If the training example if new then we will update the parameter with the new example
        #If it is already used once to update the parameter in this epoch, pass
        try:
            train[(state.ID, action)]
        except:
            Q_pred_new_state = np.dot(Basis,theta)
            max_q = max(Q_pred_new_state)
            # m_i = Q_pred_new_state.index(max_q)
            # action_max = action_order[m_i]

            Qmatrix_previous = Qmatrix.copy() #Store Qmatrix's previous values

            if math.isnan(reward + max_q):
                print('Something is wrong!')

            Qmatrix.iloc[state.ID][action] = reward + max_q
            QalphaMatrix.iloc[state.ID][action] += 1

            bas = np.asarray(phi_sa[(state.ID, action)])
            # sc = StandardScaler()
            # bas = sc.fit_transform(bas)

            #this is the optimal Q value calc from value iteration
            #Checking the target for states except 0 wouldn't work, because the other states might be different
            #If an action that is different is taken state 1 is completely different from the target's state 1
            q_pred = np.dot(bas, theta)
            # if math.isnan(q_pred):
            #     print('There is something wrong')
            target, corresp_id, q_column = funcs2.Qtarget(state,id_dict,action,q_column, state_mapper, new_state, Qmatrix)
            if math.isnan(target):
                print('NNAN')
            q_column.set_value(str((corresp_id, action)), 'q_pred', q_pred)
            q_optimal[(state.ID, action)] = target #Later you need this mapping
            q_check[(state.ID, action)]['Epoch {}'.format(ep)] = target

            #target = Qmatrix.iloc[state.ID][action]
            error = target - q_pred
            theta_next = theta + (step_size * (error * bas))

            # Record the newly found training example
            train[(state.ID, action)] = 1
            example_no += 1
            s.append(Qmatrix.iloc[0][0])
            #print('State:',state.ID,'Iteration {} - (s,a): ({},{}) - Error:'.format(example_no, state.ID, action), error)
            theta = theta_next


    ############ TO DO MAKE A DIFF ERROR CALC. MAKE THE FINAL PRED vVALUES
    ### MERGE IT WITH THE CORRESPONDING TARGET VALUES - GET THE DIFF THEN ABS SUM IT
    phi_df = pd.DataFrame.from_dict(phi_sa, orient='index')
    pred_vec = phi_df.dot(theta)
    pred_vec = pred_vec.to_frame()
    pred_vec.columns = ['q_pred']
    q_opt_df = pd.DataFrame.from_dict(q_optimal, orient='index')
    q_opt_df.columns = ['q_optimal']
    pred_opt_df = pred_vec.merge(q_opt_df, how='right', left_index = True, right_index = True)

    pred_opt_df['Epoch {}'.format(ep)] = pred_opt_df['q_pred'] - pred_opt_df['q_optimal']
    #total_error = abs(pred_opt_df['Epoch {}'.format(ep)]).sum()
    total_error = sum(pred_opt_df['Epoch {}'.format(ep)]**2)
    print('Total absolute error in epoch {}:'.format(ep), total_error)
    #step_size -= (step_size/(n_epochs-1))

#Extract the optimal policy: action for each state
valid_state_num = len(state_dict)/5
policy = funcs2.extractPolicy(Qmatrix, valid_state_num)