
import pandas as pd
import numpy as np
import funcs2
import state2 as st
import random
import networkx as nx
import SNEBC
from copy import copy
import random

random.seed(42)

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

def sample(first_state, actions, supply_nodes, resource, Qmatrix, Schedule, Q_alphaMatrix, G_restored, G2, G, EdgeList, reachable_nodes,
           ActionList, dist, phi_sa, total_debris, total_supply, explored_states, state_dict, id_dict, id_counter, betw_centrality_service,
           betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp):

    #These parameters are very likely that they are not being used
    epsilon = 0.3
    rule = 'glie'
    e=1
    T=10
    # n_episodes = 10000
    # alpha = 0.1
    n_nodes = len(G.nodes)
    Cost = np.zeros((n_nodes, n_nodes))

    # debris_feature = total_debris - sum(first_state.rem_debris)  # This is the debris cleared until now
    # demand_feature = total_supply - sum(first_state.rem_supply)  # This is the total demand satisfied until now

    #Choose action
    action = first_state.choose_action(epsilon, Qmatrix, actions, Schedule, rule, T, Q_alphaMatrix, e)

    Schedule.append(action)

    ## Vertex collapse - condense the network
    # For large sized instances calculating sp can be hard
    #betw centrality is just used to create the basis for s,a - if already calculated then don't redo it
    #Its not in constructfeatures function because it had to be done before updating G_restored etc
    try:
        betw_centrality_service[first_state.ID]
    except:
        betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp = SNEBC.BC_calcs(
            G_restored, first_state, G2,
            EdgeList, betw_centrality_service, betw_centrality_regular, betw_centrality_debris,
            betw_centrality_regular_sp)

    ######### Realize the new state and get its information #########
    #################################################################
    #Find where that action leads - how the graph changes
    new_node, discovered_nodes = funcs2.get_newReachableNode(set(reachable_nodes), action, ActionList, G_restored, G2)

    #Update the action list by adding the new_node's connections
    if new_node is not None:
        funcs2.updateActions(new_node, actions, ActionList, G)

        #Find from which supply locations the new_node is accessible
        connected_supply = first_state.establishSupplyConnection(new_node, G_restored)

        #If the newly found node connects supplies - then supply transfer
        if len(connected_supply) > 1:
            first_state.transferSupply(connected_supply)

    #Get the resource usage and update remaining debris amounts
    new_rem_debris, resource_usage = first_state.updateDebris(action)

    period_before = funcs2.getPeriod(first_state.cum_resource, resource)
    first_state.cum_resource = first_state.cum_resource + resource_usage


    #Update the planning horizon and resource amounts
    period = funcs2.getPeriod(first_state.cum_resource, resource)

    #Construct features
    #Not yet demand is realized and not allocated yet
    phi_sa, new_phi_check = funcs2.constructfeatures(first_state, action, phi_sa, ActionList, period,
                      resource_usage,  total_debris,  betw_centrality_service[first_state.ID], period_before, total_supply,
                                                     betw_centrality_regular[first_state.ID], betw_centrality_debris[first_state.ID], betw_centrality_regular_sp[first_state.ID])

    # First realize demand then allocate supply immediately
    new_rem_demand, new_rem_supply, satisfied_demand, dem = first_state.realizeDemand(new_node, dist, connected_supply, G_restored, Cost, reachable_nodes)

    ###### ---------------------- 12 --------------------------####
    if new_phi_check == 1:
        if dem > 0:
            # For now the mean distributions are the same
            # But if demand nodes have diff dist then mean_dist is going to be the mean of each dist
            mean_dist = 3
            phi_sa[(first_state.ID, action)].append(mean_dist)
        else:
            phi_sa[(first_state.ID, action)].append(0)
    ###### -----------------------------------------------------------------------##################

    reachable_nodes = discovered_nodes

    #Calculate the reward to switch to the next state
    reward = funcs2.getReward(period, satisfied_demand)

    #Create the new state
    new_state = st.State(new_rem_debris, new_rem_supply, new_rem_demand, first_state.cum_resource, None)


    #Get its index
    id_counter, id_dict = new_state.getStateIndex(id_counter, id_dict)

    state_dict[(new_state.ID, 'demand')] = copy(new_state.rem_demand)
    state_dict[(new_state.ID, 'debris')] = copy(new_state.rem_debris)
    state_dict[(new_state.ID, 'supply')] = copy(new_state.rem_supply)
    state_dict[(new_state.ID, 'period')] = copy(period)
    state_dict[(new_state.ID, 'resource')] = copy(new_state.cum_resource)

    if new_state.ID not in explored_states and sum(new_state.rem_demand)>0:
        explored_states.append(new_state.ID)

    return phi_sa, action,id_counter, new_state, reward, period, actions, betw_centrality_service,  \
           betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp, reachable_nodes


def new_state_basis(new_state, phi_sa, ActionList, cum_resource, G_restored, EdgeList, G2, total_debris, actions, betw_centrality_service, total_supply,
                    betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp, reachable_nodes, resource ):

    BasisMatrix = []
    done_actions = [i for i, val in enumerate(new_state.rem_debris) if val ==0] #Not cleared roads
    eligible_actions = actions - set(done_actions)

    for action in eligible_actions:

        try:
            phi_sa[(new_state.ID,action)]
        except:
            try:
                betw_centrality_service[new_state.ID]
            except:
                betw_centrality_service, betw_centrality_regular, betw_centrality_debris, betw_centrality_regular_sp = SNEBC.BC_calcs(
                    G_restored, new_state, G2,
                    EdgeList, betw_centrality_service, betw_centrality_regular, betw_centrality_debris,
                    betw_centrality_regular_sp)

            resource_usage = new_state.rem_debris[action]
            period = funcs2.getPeriod(cum_resource + resource_usage, resource)
            period_before = funcs2.getPeriod(cum_resource, resource)

            # Find the new_node the action leads to
            ed = [edge for edge, edge_id in ActionList.items() if edge_id == action][0]
            for node in reachable_nodes:
                for n in range(2):
                    if node == ed[n]:
                        new_node = ed[abs(1 - n)]

            phi_sa, new_phi_check = funcs2.constructfeatures(new_state, action, phi_sa, ActionList, period,
                                              resource_usage, total_debris, betw_centrality_service[new_state.ID], period_before, total_supply,
                                                             betw_centrality_regular[new_state.ID],betw_centrality_debris[new_state.ID], betw_centrality_regular_sp[new_state.ID])

            ###### ---------------------- 12 --------------------------####

            if (new_state.rem_demand[new_node] != 0) & (new_node not in reachable_nodes):  # new_node is a demand node and not discovered before
                dem = 1
                if new_phi_check ==1:
                    mean_dist = 3
                    phi_sa[(new_state.ID, action)].append(mean_dist)
            else:
                dem = 0
                if new_phi_check == 1:
                    phi_sa[(new_state.ID, action)].append(0)

            ###### -----------------------------------------------------------------------##################

        BasisMatrix.append(np.asarray(phi_sa[(new_state.ID, action)]))

    return phi_sa, eligible_actions, BasisMatrix, betw_centrality_service,  betw_centrality_regular, \
           betw_centrality_debris, betw_centrality_regular_sp