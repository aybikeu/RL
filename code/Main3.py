
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
from collections import defaultdict
from nested_dict import nested_dict
import functools




random.seed(42)

objdict = {}

#For the initial small-sized instance

#G = nx.read_edgelist('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/GridNetwork_1.csv',delimiter=',',  nodetype=int, data=(('debris',float),))
G = nx.read_edgelist('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/GridNetwork_1.csv',delimiter=',',  nodetype=int, data=(('debris',float),))
EdgeList = G.edges()
ActionList = dict(zip(EdgeList, range(len(EdgeList))))

a = list(G.edges.data('debris'))
initial_debris = list(zip(*a)[2])

#df_node_data = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/GridNetwork_1_sd.csv', header=None)
df_node_data = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/GridNetwork_1_sd.csv', header=None)

initial_supply = df_node_data[0].tolist() # for all nodes
demand_indicator= df_node_data[1].tolist() # for all nodes

total_debris = sum(initial_debris)
total_supply = sum(initial_supply)

# Define the demand distribution - the same for all demand points
max_demand = 5
xk = np.arange(1, max_demand+1)
pk = [0.1 , 0.2 , 0.4, 0.2, 0.1]
dist = stats.rv_discrete(name='dist', values=(xk, pk))
mean_dist = dist.mean()

#Integer demand values - simply rounded DOWN
initial_demand = map(lambda x: (x*mean_dist).round(0), demand_indicator) #For now equate the rem_demand of demand nodes to their exp value
supply_nodes = [i for i,p in enumerate(initial_supply) if p>0]

resource = 1 #for the second instance
#resource = 5

n_edges = len(initial_debris)
n_actions = n_edges #All the edges blocked with debris are the possible actions
n_nodes = len(initial_supply)

#Parameters
n_episodes = 300000
state_dict = {}

#Parameters to set the id's of states
id_dict = {}
id_counter = 0

r_sas = nested_dict()
p_sas = nested_dict()

phi_sa = {}
betw_centrality_service = {}
betw_centrality_regular = {}
betw_centrality_debris = {}
betw_centrality_regular_sp = {}

explored_states = []

state_dict[(0, 'demand')] = initial_demand
state_dict[(0, 'debris')] = initial_debris
state_dict[(0, 'supply')] = initial_supply
state_dict[(0, 'period')] = 1
state_dict[(0, 'resource')] = 0


initial_state = st.State(initial_debris, initial_supply, initial_demand, 0, None)
id_counter, id_dict = initial_state.getStateIndex(id_counter, id_dict)
actions = funcs2.initializeActionSpace(supply_nodes, G, ActionList)
explored_states.append(initial_state.ID)


for e in range(int(n_episodes)):


    # Initialize the environment
    #remaining_demand = True
    Schedule = [] #Initialization - these roads are as if they are cleared before

    reachable_nodes = set(supply_nodes)
    rem_resource = resource
    period = 1
    G_restored = nx.Graph()
    G_restored.add_nodes_from(range(n_nodes))
    leftover_demand = []
    Cost = np.zeros((n_nodes, n_nodes)) #Cost matrix for the transportation problem - for supply allocation to demand
    objective = 0
    #total_resource_usage = 0

    sas_vec = []

    G2 = G.copy()

    #Initialize the state and the action space
    #Set up the initial status of the environment
    first_state_id = random.choice(explored_states)
    remaining_demand = sum(state_dict[(first_state_id, 'demand')])>0
    first_state = st.State(state_dict[(first_state_id, 'debris')], state_dict[(first_state_id, 'supply')], state_dict[(first_state_id, 'demand')],
                           state_dict[(first_state_id, 'resource')], first_state_id)

    #This is not the ordered schedule but rather the roads cleared until this state
    cleared_roads = [i for i, val in enumerate(state_dict[(first_state_id, 'debris')]) if val == 0]
    Schedule.extend(cleared_roads)


    #G2 is the original graph whereas G_disrupted is the graph to be constructed with the cleared roads
    for cr in Schedule:
        ed = [edge for edge, edge_id in ActionList.items() if edge_id == cr][0]
        G2[ed[0]][ed[1]]['debris'] = 0
        G_restored.add_edge(ed[0], ed[1])

    reachable_nodes = [] #list for reachable nodes
    for s in supply_nodes:
        reachable_nodes.extend(list(nx.dfs_preorder_nodes(G_restored, s)))

    actions = funcs2.initializeActionSpace(reachable_nodes, G, ActionList)  # actions are the admissable action indices corresponding in ActionList

    ############sas_vec.append(first_state.ID)
    while remaining_demand:

        #Choose action
        eligible_actions = actions - set(Schedule)
        action = random.choice(list(eligible_actions))
        Schedule.append(action)

        ## Vertex collapse - condense the network
        #For large sized instances calculating sp can be hard
        try:
            betw_centrality_service[first_state.ID]
        except:
            betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp = SNEBC.BC_calcs(G_restored, first_state, G2,
                                                        EdgeList, betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp)

        ######### Realize the new state and get its information #########
        #################################################################
        #Find where that action leads - how the graph changes
        new_node, discovered_nodes = funcs2.get_newReachableNode(set(reachable_nodes), action, ActionList, G_restored, G2)

        #Update the action list by adding the new_node's connections
        funcs2.updateActions(new_node, actions, ActionList, G)

        #Find from which supply locations the new_node is accessible
        connected_supply = first_state.establishSupplyConnection(new_node, G_restored)

        #If the newly found node connects supplies - then supply transfer
        if len(connected_supply) > 1:
            first_state.transferSupply(connected_supply)

        #Get the resource usage and update remaining debris amounts
        new_rem_debris, resource_usage = first_state.updateDebris(action)

        first_state.cum_resource = first_state.cum_resource + resource_usage

        period_before = period
        #Update the planning horizon and resource amounts
        period = funcs2.getPeriod(first_state.cum_resource, resource)

        #Construct features
        #Before realizing demand so that you can omit the effect of the action taken & realization
        phi_sa, new_phi_check = funcs2.constructfeatures(first_state, action, phi_sa, ActionList, period,
                                          resource_usage, total_debris, betw_centrality_service[first_state.ID], period_before, total_supply,
                                                         betw_centrality_regular[first_state.ID], betw_centrality_debris[first_state.ID], betw_centrality_regular_sp[first_state.ID])

        # Realize demand then allocate supply immediately
        new_rem_demand, new_rem_supply, satisfied_demand, dem = first_state.realizeDemand(new_node, dist, connected_supply, G_restored, Cost, reachable_nodes)

        #####----------------------------- 12 -----------------------------------------------#########
        ####### ------------- Information gain
        # If the node reached is a NEW (not realized before) demand node - binary
        if new_phi_check == 1:
            if dem > 0:
                # For now the mean distributions are the same
                # But if demand nodes have diff dist then mean_dist is going to be the mean of each dist
                phi_sa[(first_state.ID, action)].append(mean_dist)
            else:
                phi_sa[(first_state.ID, action)].append(0)
        ###### -----------------------------------------------------------------------##################

        reachable_nodes = discovered_nodes

        #Calculate the reward to switch to the next state
        reward = funcs2.getReward(period, satisfied_demand)
        objective += reward

        #Create the new state
        new_state = st.State(new_rem_debris, new_rem_supply, new_rem_demand, first_state.cum_resource, None)


        #Get its index
        id_counter, id_dict = new_state.getStateIndex(id_counter, id_dict)

        r_sas[first_state.ID][action][new_state.ID] = reward
        if dem==0: #dem checks if a new demand node is realized
            p_sas[first_state.ID][action][new_state.ID] = 1
        else:
            p_sas[first_state.ID][action][new_state.ID] = pk[dem-1]

        state_dict[(new_state.ID, 'demand')] = copy(new_state.rem_demand)
        state_dict[(new_state.ID, 'debris')] = copy(new_state.rem_debris)
        state_dict[(new_state.ID, 'supply')] = copy(new_state.rem_supply)
        state_dict[(new_state.ID, 'period')] = copy(period)
        state_dict[(new_state.ID, 'resource')] = copy(new_state.cum_resource)

        if new_state.ID not in explored_states and sum(new_state.rem_demand)>0:
            explored_states.append(new_state.ID)

        #Qmatrix = funcs2.updateQ(Qmatrix, Q_alphaMatrix, action, first_state, new_state, reward, n_episodes,alpha)

        #Check termination
        if sum(new_rem_demand)==0: #All the demand is satisfied
            remaining_demand = False
        else:
            first_state = new_state


df_basis= pd.DataFrame(data=phi_sa.values(),index=phi_sa.keys(),columns=['Intercept','SNEBC','resource','Rem. demand',
                                                                         'Rem. Debris', 'Satisfied demand','Exp period before',
                                                                          'Exp period after','BC_debris','BC_regular','BC_regular_sp','IG_1'])
df_basis.sort_index(inplace=True)


df_basis.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/basis_100kVI_INS4.csv', sep=',')



df_pr=pd.DataFrame(columns=('probability','reward','s','a','s_prime'),dtype=float)
for s, v1 in p_sas.items():
    for a, v2 in v1.items():
        for s_prime, v3 in v2.items():
            x = pd.DataFrame({'probability': v3, 'reward': r_sas[s][a][s_prime], 's': s, 'a':a, 's_prime':s_prime}, index=['('+str(s)+','+str(a)+','+str(s_prime)+')'])
            df_pr = df_pr.append(x)

df_pr.set_index(['s','a'], inplace=True)
df_pr.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/pr_for_INS4.csv', sep=',')

###################################### SECOND PART FINDING THE OPTIMAL Q_VALUES ###########################
df_pr = pd.read_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/pr_for_INS4.csv', sep=',')

epsilon = 0.1
convergence_check=False

#n_states is 862 for Instance 2 - 1053 for instance 1
n_states = int(df_pr['s_prime'].sort_values(ascending=False).values[0] + 1)
n_actions = 17

n_sas = df_pr.shape[0] #number of all (s, a, s_prime)

Q_T = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)
Q_Tnext = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)

#while convergence_check==False:
for _ in range(20):
    for s in range(n_states):
        for a in range(n_actions):
            df_slice = df_pr.query('s== {} & a=={}'.format(s,a))
            added_value = 0
            for _, r in df_slice.iterrows():
                    s_p = int(r['s_prime'])
                    reward = r['reward']
                    pr = r['probability']
                    added_value = added_value + (pr * (reward + Q_T.iloc[s_p].max()))

            Q_Tnext.iloc[s][a] = added_value

    Q_diff = Q_Tnext - Q_T
    Q_T = Q_Tnext.copy()

if Q_diff.sum().sum() <= epsilon * 3000:
    convergence_check = True

Q_Tnext.to_csv('C:/Users/ulusan.a/Desktop/RL_rep/RL/data_files/Q_optimalVI_INS4.csv',sep=',')