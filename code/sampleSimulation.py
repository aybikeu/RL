
import pandas as pd
import numpy as np
import funcs2
import state2 as st
import random
import networkx as nx
import SNEBC
from copy import copy

def buildEnvironment(explored_states, state_dict, G, G2, G_disrupted, ActionList, supply_nodes):

    Schedule = []
    first_state_id = random.choice(explored_states)
    remaining_demand = sum(state_dict[(first_state_id, 'demand')]) > 0
    first_state = st.State(state_dict[(first_state_id, 'debris')], state_dict[(first_state_id, 'supply')],
                           state_dict[(first_state_id, 'demand')],
                           state_dict[(first_state_id, 'resource')], first_state_id)

    # This is not the ordered schedule but rather the roads cleared until this state
    cleared_roads = [i for i, val in enumerate(state_dict[(first_state_id, 'debris')]) if val == 0]
    Schedule.extend(cleared_roads)

    for cr in cleared_roads:
        ed = [edge for edge, edge_id in ActionList.items() if edge_id == cr][0]
        G2[ed[0]][ed[1]]['debris'] = 0
        G_disrupted.add_edge(ed[0], ed[1])

    reachable_nodes = []  # list for reachable nodes
    for s in supply_nodes:
        reachable_nodes.extend(list(nx.dfs_preorder_nodes(G_disrupted, s)))

    actions = funcs2.initializeActionSpace(reachable_nodes, G,
                                           ActionList)  # actions are the admissable action indices corresponding in ActionList

    return first_state, actions, Schedule, reachable_nodes

def sample(first_state, actions, supply_nodes, resource, Qmatrix, Schedule, Q_alphaMatrix, G_disrupted, G2, G, EdgeList, reachable_nodes,
           ActionList, dist, phi_sa, total_debris, total_supply, explored_states, state_dict, id_dict, id_counter):

    #These parameters are very likely that they are not being used
    epsilon = 0.3
    rule = 'glie'
    e=1
    T=10
    n_episodes = 10000
    alpha = 0.1
    n_nodes = len(G.nodes)
    Cost = np.zeros((n_nodes, n_nodes))

    # debris_feature = total_debris - sum(first_state.rem_debris)  # This is the debris cleared until now
    # demand_feature = total_supply - sum(first_state.rem_supply)  # This is the total demand satisfied until now

    #Choose action
    action = first_state.choose_action(epsilon, Qmatrix, actions, Schedule, rule, T, Q_alphaMatrix, e)

    Schedule.append(action)

    ## Vertex collapse - condense the network
    # For large sized instances calculating sp can be hard
    # G_collapsed = nx.condensation(G_disrupted.to_directed())
    # demand_collapsed , supply_collapsed, G_collapsed = funcs2.fixcondensation(G_collapsed, first_state.rem_demand, first_state.rem_supply, G2)
    # betw_nodes = SNEBC.SNEBC(G_collapsed, demand_collapsed, supply_collapsed, weight='debris')
    # betw_nodes_uncollapsed = SNEBC.uncollapse(betw_nodes, G_collapsed)
    # betw_edges = SNEBC.convert2edge(betw_nodes_uncollapsed, EdgeList)

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
                                      G2, new_node, ActionList, period, resource_usage,
                                      satisfied_demand, G_disrupted, total_debris, total_supply, EdgeList)

    reachable_nodes = discovered_nodes

    #Calculate the reward to switch to the next state
    reward = funcs2.getReward(period, satisfied_demand)

    #Create the new state
    new_state = st.State(new_rem_debris, new_rem_supply, new_rem_demand, first_state.cum_resource, None)


    #Get its index
    id_counter, id_dict = new_state.getStateIndex(id_counter, id_dict)

    # r_sas[first_state.ID][action][new_state.ID] = reward
    # if dem==0:
    #     p_sas[first_state.ID][action][new_state.ID] = 1
    # else:
    #     p_sas[first_state.ID][action][new_state.ID] = pk[dem-1]
    state_dict[(new_state.ID, 'demand')] = copy(new_state.rem_demand)
    state_dict[(new_state.ID, 'debris')] = copy(new_state.rem_debris)
    state_dict[(new_state.ID, 'supply')] = copy(new_state.rem_supply)
    state_dict[(new_state.ID, 'period')] = copy(period)
    state_dict[(new_state.ID, 'resource')] = copy(new_state.cum_resource)

    if new_state.ID not in explored_states and sum(new_state.rem_demand)>0:
        explored_states.append(new_state.ID)

    #Qmatrix = funcs2.updateQ(Qmatrix, Q_alphaMatrix, action, first_state, new_state, reward, n_episodes,alpha)


    #Q_sa = reward + Qmatrix.iloc[second_state.ID].max()

    #construct features for the new_state so that you can calculate the predicted Q_values
    phi_sa = funcs2.constructfeatures(new_state, action, phi_sa, discovered_nodes, reachable_nodes,
                                      G2, new_node, ActionList, period, resource_usage,
                                      satisfied_demand, G_disrupted, total_debris, total_supply, EdgeList)


    return Qmatrix, action, phi_sa, id_counter, new_state, reward