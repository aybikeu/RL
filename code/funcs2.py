
import numpy as np
from operator import itemgetter
import networkx as nx
from pulp import *
import pandas as pd
import math
import SNEBC

def initializeActionSpace (reachable_nodes, G, ActionList):

    actions = set()
    for i in reachable_nodes:
        neigh = G.neighbors(i)
        edges = [sorted((i, e)) for e in neigh]
        for f in edges:
            actions.add(ActionList[tuple(f)])

    return actions

def get_newReachableNode(reachable_nodes, action, ActionList, G_collapsed, G2):
    ed = [edge for edge, edge_id in ActionList.items() if edge_id == action][0]

    G_collapsed.add_edge(ed[0],ed[1])

    for node in reachable_nodes: #There is always going to be 1 node newly discovered - because the network is always composed of debris edges
        for n in range(2):
            if node == ed[n]:
                new_node = ed[abs(1 - n)]

    # if new node is not in reachable set - add it to the list
    # Because an edge both end nodes that are reachable can be picked as action
    new_reachable_nodes = reachable_nodes.copy()
    new_reachable_nodes.add(new_node)

    G2[ed[0]][ed[1]]['debris']=0

    return new_node, new_reachable_nodes

def updateActions(new_node, actions, ActionList, G):

    #Update the action set by first realizing newly reached node

    neigh = G.neighbors(new_node)
    edges = [sorted((new_node, e)) for e in neigh]

    #Add the new admissable actions to the list
    #This set might have the actions that are selected before as well
    for f in edges:
        actions.add(ActionList[tuple(f)])

    #Do not remove the previously done actions
    #You don't want to restrict the agent but to make him learn that it shouldn't take the actions he took before




def benefitFunction (constant, Lambda, t):
    benefit = constant *np.exp(-Lambda*t)
    return benefit


def updateQ (Q, Q_a,  action, first_state, second_state, reward, n_eps,alpha):
    q_sa = Q.iloc[first_state.ID][action]

    alpha = 1.0/ (1+Q_a.iloc[first_state.ID][action])
    #alpha = np.log(1+ Q_a.iloc[first_state.ID][action]) / (1 + Q_a.iloc[first_state.ID][action])
    q_sa = (q_sa * (1-alpha)) + (alpha * (reward + Q.iloc[second_state.ID].max()))
    Q.iloc[first_state.ID][action]=q_sa
    Q_a.iloc [first_state.ID][action] = Q_a.iloc[first_state.ID][action] + 1
    return Q

def updatePredQ(Qmatrix, first_state, new_state, reward, theta, phi_sa, n_actions):

    max_q = -1
    for a in range(n_actions):
        basis_newstate = phi_sa[(new_state.ID, a)]
        bas = np.asarray(basis_newstate)
        basis_newstate = bas[[2, 4, 5, 6]]
        q_sa = np.dot(basis_newstate, theta)
        Qmatrix.iloc[new_state.ID][a] = q_sa

    q = reward + max_q
    return q
# def getPeriod(rem_resource, period, resource_usage, R):
#     if resource_usage > rem_resource:
#         period += 1
#         rem_re.source = R + rem_resource - resource_usage
#     else:
#         rem_resource = rem_resource - resource_usage
#     return period, rem_resource
def getPeriod (cum_resource, R):
    period = math.floor((cum_resource-0.01)/R) +1
    return period


def getReward (period, satisfied_demand):
    C = 100 #The constant in the benefit function
    L = 0.2 #Lambda parameter
    benefit_parameter = benefitFunction(C,L,period)
    reward = benefit_parameter * satisfied_demand
    return reward

#def optimalPolicy(QMatrix):


def updateCost (new_node, connected_sup, demand_nodes, G_collapsed, Cost):
    check = 0
    for potential_demand in demand_nodes:
        if nx.has_path(G_collapsed, new_node, potential_demand):
            for s in connected_sup:
                Cost[s,potential_demand]=1 #Set the connection
                check=1 #The check for the update
    return Cost, check



def supplyAllocation(costs, demand_amount , supply_amount, demand, supply):


    problem = LpProblem("Supply Allocation", LpMaximize)

    paths = [(S, D) for S in supply for D in demand]
    dvar = LpVariable.dicts("x", (supply, demand), 0, None, LpInteger)

    # Add to the objective function for the problem
    problem += lpSum([dvar[s][t] * costs[s][t] for (s, t) in paths])

    for s in supply:
        problem += lpSum([dvar[s][t] for t in demand if costs[s][t] > 0]) <= supply_amount[s]

    for t in demand:
        problem += lpSum([dvar[s][t] for s in supply if costs[s][t] > 0]) <= demand_amount[t]

    problem.writeLP('SupDem_Allocation.lp')
    problem.solve()

    cum_dem = 0
    for s in dvar.keys(): #gives the supplies
        for t in dvar[s].keys(): #gives the demand
            if dvar[s][t].varValue != None:
                cum_dem += dvar[s][t].varValue  #to find the total satisfied demand
                demand_amount[t] -= dvar[s][t].varValue
                supply_amount[s] -= dvar[s][t].varValue

    satisfied_demand = cum_dem


    return demand_amount, supply_amount, satisfied_demand

def extractPolicy(Qmatrix, num):
    opt_val = Qmatrix.max(axis=1).iloc[:num-1]
    policy = Qmatrix.idxmax(axis=1).iloc[:num-1]
    d = {'Opt Value': opt_val, 'Opt Action': policy}
    policy_df = pd.DataFrame(data =d)
    return policy_df


def fixcondensation(G_collapsed, demand, supply, G2):

    G_collapsed = G_collapsed.to_undirected()

    inv_map = {}
    for k, v in G_collapsed._node.iteritems():
        val = list(v['members'])

        for vv in val:
            inv_map[vv] = k



    npsup = np.asarray(supply)
    npdem = np.asarray(demand)

    demandvec = []
    supplyvec = []
    for n,d in G_collapsed.nodes(data=True):

        comp_nodes = list(d['members'])
        tot_sup = npsup[comp_nodes].sum()
        tot_dem = npdem[comp_nodes].sum()

        G_collapsed.nodes[n]['supply'] = tot_sup
        G_collapsed.nodes[n]['demand'] = tot_dem

        demandvec.append(tot_dem)
        supplyvec.append(tot_sup)

    for f,t,d in G2.edges(data=True):

        d = d['debris']

        mapped_f = inv_map[f]
        mapped_t = inv_map[t]

        if mapped_f != mapped_t:
            try: #There might be multiple edges
                current_weight = G_collapsed[mapped_f][mapped_t]['debris']
                if d < current_weight:
                    G_collapsed[mapped_f][mapped_t]['debris'] = d

            except KeyError:
                G_collapsed.add_edge(mapped_f, mapped_t)
                G_collapsed[mapped_f][mapped_t]['debris'] = d

    return demandvec, supplyvec, G_collapsed

def constructfeatures(first_state, action, phi_sa, ActionList, period,
                      resource_usage,  total_debris,  betw_edges ):
    try: # if the basis is already constructed, no need to do the same calculations again
        phi_sa[(first_state.ID, action)]
    except KeyError:
        phi_sa[(first_state.ID, action)] = []

        ###--------------------------- 1 ----------------------------####

        # Add SNEBC measure for s,a pair
        act = [key for key, value in ActionList.items() if value == action][0]
        phi_sa[(first_state.ID, action)].append(betw_edges[act])

        ###--------------------------- 2 ---------------------------###
        #amount of resource usage (debris amount) on the action(road)
        phi_sa[(first_state.ID, action)].append(resource_usage)

        ###--------------------------- 3 ---------------------------### THIS IS A STATE PROPERTY THOUGH
        # Even if the new reached node is not a demand point it can be connected to other demand
        # via unblocked roads - hence should be taken into account
        # The amount of satisfied demand is going to change - hence binary makes more sense
        # If we had different prob distributions then we could've put the expected val of the demand dist too
        #
        # if satisfied_demand > 0:
        #     phi_sa[(first_state.ID, action)].append(satisfied_demand)
        # else:
        #     phi_sa[(first_state.ID, action)].append(0)

        phi_sa[(first_state.ID, action)].append(sum(first_state.rem_demand))  # Total realized demand
        #phi_sa[(first_state.ID, action)].append(np.count_nonzero(np.asarray(first_state.rem_demand)) * 3)  # total expected rem_demand

        ###--------------------------- 4 ---------------------------###
        debris_feature = total_debris - sum(first_state.rem_debris)  # This is the debris cleared until now
        phi_sa[(first_state.ID, action)].append(debris_feature)  # Total cleared debris until now

        ###--------------------------- 5 ---------------------------###

        # demand_feature = total_supply - sum(first_state.rem_supply)  # This is the total demand satisfied until now
        # phi_sa[(first_state.ID, action)].append(demand_feature)  # Total satisfied demand until now

        ###--------------------------- 6 ---------------------------###
        phi_sa[(first_state.ID, action)].append(np.exp(-0.2 * period))


    return phi_sa

