
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

#T_values = [1, 5, 10, 20]
#T_values = [1000,300,100]
T_values=[100,300,1000]
#for temp in T_values:
#For the initial small-sized instance
EdgeList = [(0,1), (0,2), (1,2), (1,3),
            (2,3), (2,4), (3,4)]

EdgeListWeighted = [(0,1,1), (0,2,3), (1,2,3), (1,3,2),
            (2,3,1), (2,4,3), (3,4,1)]

ActionList = dict(zip(EdgeList, range(len(EdgeList))))

G = nx.Graph()
G.add_weighted_edges_from(EdgeListWeighted,weight='debris')

initial_debris = [1,3,3,2,1,3,1] #for all edges
initial_supply = [5,0,0,0,5] # for all nodes
demand_indicator= [0,0,1,1,0] # for all nodes

total_debris = sum(initial_debris)
total_supply = sum(initial_supply)

#This graph is going to be modified with each action
G2 = nx.Graph()
G2.add_weighted_edges_from(EdgeListWeighted, weight='debris')

# Define the demand distribution - the same for all demand points
max_demand = 5
xk = np.arange(1, max_demand+1)
pk = [0.1 , 0.2 , 0.4, 0.2, 0.1]


#pk = [0.8, 0.05 , 0.05, 0.09, 0.01]
dist = stats.rv_discrete(name='dist', values=(xk, pk))

mean_dist = dist.mean()

#Integer demand values - simply rounded DOWN
initial_demand = map(lambda x: (x*mean_dist).round(0), demand_indicator) #For now equate the rem_demand of demand nodes to their exp value
actionSpace = [] #For the initial action space
supply_nodes = [i for i,p in enumerate(initial_supply) if p>0]

resource = 5


n_edges = len(initial_debris)
n_actions = n_edges #All the edges blocked with debris are the possible actions
n_nodes = len(initial_supply)



# Figure out how many different combinations for states
n_debriscomb = 2**n_edges
n_supplycomb = reduce((lambda x,y: x*y), [j+1 for j in initial_supply])
#n_demandcomb = 2**(len([i for i in demand_indicator if i >0]))
n_demandcomb = reduce((lambda x,y: x*y), [j+1 for j in map(lambda x: x*max_demand, demand_indicator)])

n_states = n_debriscomb * n_supplycomb * n_demandcomb

#Initialize all Q(s,a) values to 0
Qmatrix = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)


#Parameters
gamma = 1 # We don't define a discount value
#n_episodes = 1E3 # A big number
n_episodes = 50000
epsilon = 0.3
temp = 10 # if greedy is chosen, dummy value assigned to temp
T = temp #Temperature parameter of boltzmann
alpha = 1

#To decrease alpha value based on the number of times s,a pair is visited
Q_alphaMatrix = pd.DataFrame(data=0,
                       index=range(n_states),
                       columns=range(n_actions),
                       dtype=float)

# State Action State
sas_dict = {}
#rule = 'greedy' #Indicates the action selection rule
#rule = 'boltzmann'
rule = 'glie'

state_dict = {}

#Parameters to set the id's of states
id_dict = {}
id_counter = 0

r_sas = nested_dict()
p_sas = nested_dict()

complete_schedule={}
 #Just a list to keep track of the selected actions

#Keep track of the objective
obj_list=[]
obj_list1=[]
obj_list2=[]
obj_list3=[]
obj_list4=[]
obj_list33=[]

phi_sa = {}

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


    #alpha = log(e+1)/(e+1)

    # Initialize the environment
    #remaining_demand = True
    Schedule = []
    reachable_nodes = set(supply_nodes)
    rem_resource = resource
    period = 1
    G_disrupted = nx.Graph()
    G_disrupted.add_nodes_from(range(n_nodes))
    leftover_demand = []
    Cost = np.zeros((n_nodes, n_nodes)) #Cost matrix for the transportation problem - for supply allocation to demand
    objective = 0
    #total_resource_usage = 0

    sas_vec = []

    G2 = nx.Graph()
    G2.add_weighted_edges_from(EdgeListWeighted, weight='debris')

    #Initialize the state and the action space
    # first_state = st.State(initial_debris, initial_supply, initial_demand, total_resource_usage, None)
    # id_counter, id_dict = first_state.getStateIndex(id_counter, id_dict)

    #Set up the initial status of the environment
    first_state_id = random.choice(explored_states)
    #first_state_id = 0
    remaining_demand = sum(state_dict[(first_state_id, 'demand')])>0
    first_state = st.State(state_dict[(first_state_id, 'debris')], state_dict[(first_state_id, 'supply')], state_dict[(first_state_id, 'demand')],
                           state_dict[(first_state_id, 'resource')], first_state_id)

    #This is not the ordered schedule but rather the roads cleared until this state
    cleared_roads = [i for i, val in enumerate(state_dict[(first_state_id, 'debris')]) if val == 0]
    Schedule.extend(cleared_roads)

    for cr in cleared_roads:
        ed = [edge for edge, edge_id in ActionList.items() if edge_id == cr][0]
        G2[ed[0]][ed[1]]['debris'] = 0
        G_disrupted.add_edge(ed[0], ed[1])

    reachable_nodes = [] #list for reachable nodes
    for s in supply_nodes:
        reachable_nodes.extend(list(nx.dfs_preorder_nodes(G_disrupted, s)))

    actions = funcs2.initializeActionSpace(reachable_nodes, G, ActionList)  # actions are the admissable action indices corresponding in ActionList

    sas_vec.append(first_state.ID)
    while remaining_demand:

        ########## Create some state related features before an action is taken and the next state is reached
        debris_feature = total_debris - sum(first_state.rem_debris)  # This is the debris cleared until now
        demand_feature = total_supply - sum(first_state.rem_supply) #This is the total demand satisfied until now


        #Choose action
        action = first_state.choose_action(epsilon, Qmatrix, actions, Schedule, rule, T, Q_alphaMatrix, e)

        Schedule.append(action)
        sas_vec.append(action)

        ## Vertex collapse - condense the network
        # For large sized instances calculating sp can be hard
        G_collapsed = nx.condensation(G_disrupted.to_directed())
        demand_collapsed , supply_collapsed, G_collapsed = funcs2.fixcondensation(G_collapsed, first_state.rem_demand, first_state.rem_supply, G2)

        betw_nodes = SNEBC.SNEBC(G_collapsed, demand_collapsed, supply_collapsed, weight='debris')
        betw_nodes_uncollapsed = SNEBC.uncollapse(betw_nodes, G_collapsed)
        betw_edges = SNEBC.convert2edge(betw_nodes_uncollapsed, EdgeList)

        ######### Realize the new state and get its information #########
        #################################################################
        #Find where that action leads - how the graph changes
        new_node, discovered_nodes = funcs2.get_newReachableNode(set(reachable_nodes), action, ActionList, G_disrupted, G2)

        #Update the action list by adding the new_node's connections
        funcs2.updateActions(new_node, actions, ActionList, G)

        #Find from which supply locations the new_node is accessible
        connected_supply = first_state.establishSupplyConnection(new_node, G_disrupted)

        #If the newly found node connects supplies - then supply transfer
        if len(connected_supply) > 1:
            first_state.transferSupply(connected_supply)


        # First realize demand then allocate supply immediately
        new_rem_demand, new_rem_supply, satisfied_demand, dem = first_state.realizeDemand(new_node, dist, connected_supply, G_disrupted, Cost, reachable_nodes)

        #Get the resource usage and update remaining debris amounts
        new_rem_debris, resource_usage = first_state.updateDebris(action)

        first_state.cum_resource = first_state.cum_resource + resource_usage

        #Update the planning horizon and resource amounts
        period = funcs2.getPeriod(first_state.cum_resource, resource)

        #Construct features

        phi_sa = funcs2.constructfeatures(first_state, action, phi_sa, discovered_nodes, reachable_nodes,
                                          G2, new_node, ActionList, debris_feature, demand_feature, period, resource_usage,
                                          satisfied_demand, betw_edges)

        reachable_nodes = discovered_nodes



        #Calculate the reward to switch to the next state
        reward = funcs2.getReward(period, satisfied_demand)
        objective += reward

        #Create the new state
        new_state = st.State(new_rem_debris, new_rem_supply, new_rem_demand, first_state.cum_resource, None)


        #Get its index
        id_counter, id_dict = new_state.getStateIndex(id_counter, id_dict)

        r_sas[first_state.ID][action][new_state.ID] = reward
        if dem==0:
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

        sas_vec.append(new_state.ID)

        #Update the Q values
       # alpha = 1.0/math.sqrt(e+1)
        #alpha = 1.0/((e+1)**(0.51))
        #alpha=1.0/(e+1)
        Qmatrix = funcs2.updateQ(Qmatrix, Q_alphaMatrix, action, first_state, new_state, reward, n_episodes,alpha)

        #For each episode check the action sequence - for output analysis
        complete_schedule[e] = Schedule

        #Check termination
        if sum(new_rem_demand)==0: #All the demand is satisfied
            remaining_demand = False
        else:
            first_state = new_state

    #obj_list.append(objective)

    sas_dict[e]=sas_vec

    obj_list.append(Qmatrix.iloc[0].max())
    # obj_list1.append(Qmatrix.iloc[1].max())
    # obj_list2.append(Qmatrix.iloc[2].max())
    # obj_list3.append(Qmatrix.iloc[3].max())
    # obj_list4.append(Qmatrix.iloc[4].max())
    # obj_list33.append(Qmatrix.iloc[33].max())


    #Decrase the epsilon
    #epsilon -= (1.0/n_episodes)
    #alpha -= (1.0 / n_episodes)
    #T -= (T/n_episodes)

#objdict[temp] = obj_list
#obj_list=[]

#Get the best action for each state
#The state number prev calculated is just an UB on the actual state number
#there can't be anymore states than the ones calculated
valid_state_num = len(state_dict)/4
policy = funcs2.extractPolicy(Qmatrix, valid_state_num)

#y = objdict[1000]
x = range(len(obj_list))
#y=objdict[1000]
y=obj_list
#plt.plot(x,y, label='T=1000')
plt.plot(x,y, label='state 0')
#plt.legend('T=1000')
xlabel('# of episodes')
ylabel('Objective value')
# plt.plot(x,objdict[300],label='T=300')
# plt.plot(x,objdict[100],label='T=100')


# plt.plot(x,objdict[20])
plt.plot(x,obj_list1, label='state 1')
plt.plot(x,obj_list4, label='state 4')
plt.plot(x,obj_list, label='state 2')
plt.plot(x,obj_list3, label='state 3')
plt.plot(x,obj_list33, label='state 33')
plt.legend()
grid(True)
show()

counter=0
for key, val in sas_dict.items():
    if val[1]==6:
        counter+=1

policy.to_csv('opt_policy_100kVI_1R2.csv', sep=',')

#phi_sorted = sorted(phi_sa.items())
df_basis= pd.DataFrame(data=phi_sa.values(),index=phi_sa.keys(),columns=['Action Betw','resource','satisfied demand',
                                                                         'mean rem demand','actual rem demand','collected debris',
                                                                         'satisfied demand','period', 'exponential period', 'degree'])
df_basis.sort_index(inplace=True)

# qsa = []
# for ind in df_basis.index:
#     ss = ind[0]
#     aa = ind[1]
#     qsa.append(Qmatrix.iloc[ss,aa])
#
# df_basis['qval']=qsa

df_basis.to_csv('basis_100kVI_1R2.csv', sep=',')


Q_alphaMatrix.to_csv('Q_alphamatrix_R2.csv',sep=',')
Qmatrix.to_csv('Q_matrix_R2.csv',sep=',')

with open('state_info_100kVI_1R2.csv', 'wb') as myfile2:
    b = csv.writer(myfile2)
    for key,val in sorted(state_dict.items()):
    #for key, val in policy.items():
        b.writerow([key,val])
#
with open('policy_100k_VIR2.csv', 'wb') as myfile1:
    b = csv.writer(myfile1)
    for key,val in sas_dict.items():
    #for key, val in policy.items():
        b.writerow([key,val])
#
with open('obj_list_1mrandomstoc.csv', 'wb') as myfile:
    a = csv.writer(myfile,delimiter=',')
    data = y
    a.writerow(y)

df_pr=pd.DataFrame(columns=('probability','reward','s','a','s_prime'),dtype=float)
for s, v1 in p_sas.items():
    for a, v2 in v1.items():
        for s_prime, v3 in v2.items():
            x = pd.DataFrame({'probability': v3, 'reward': r_sas[s][a][s_prime], 's': s, 'a':a, 's_prime':s_prime}, index=['('+str(s)+','+str(a)+','+str(s_prime)+')'])
            df_pr = df_pr.append(x)

df_pr.set_index(['s','a'], inplace=True)

df_pr.to_csv('pr_forVIR2.csv', sep=',')


print ( "The end")