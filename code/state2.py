import random
import networkx as nx
import numpy as np
import funcs2
import copy
from scipy import stats
from operator import itemgetter

class State():
    #random.seed(42)
    def __init__(self, rem_debris, rem_supply, rem_demand, tot_resource, id):

        self.rem_debris = copy.copy(rem_debris)
        self.rem_supply = copy.copy(rem_supply)
        self.rem_demand = copy.copy(rem_demand)
        #self.actions = actionSpace
        self.cum_resource = copy.copy(tot_resource)
        self.ID = id





    def getStateIndex(self, id_counter, id_dict):
        #Mapping each state to a unique number with hash
        #state_id = hash((hash(tuple(self.rem_debris)), hash(tuple(self.rem_supply)), hash(tuple(self.rem_demand)), self.cum_resource))
        state_id = hash((hash(tuple(self.rem_debris)), hash(tuple(self.rem_supply)), hash(tuple(self.rem_demand))))
        #Updating the id dictionary
        #If you already have that key don't replace it with a new value
        if state_id not in id_dict:
            id_dict[state_id]=id_counter
            id_counter +=1

        self.ID = id_dict[state_id]

        return id_counter, id_dict


    def choose_action(self, epsilon, Qmatrix, actions, S,rule, T, Q_alphamat, episode):

        eligible_actions = actions - set(S)
        q = Qmatrix.loc[self.ID]
        if rule =='glie':

            #visit_times = Q_alphamat.iloc[self.ID].sum()
            action = random.choice(list(eligible_actions))
            # else:
            #exploration_prob = 1-(visit_times/(episode+1))
            # exploration_prob
            # maxQ= max(q)
            # #best_action = [i for i in range(len(q)) if q[i] == maxQ]
            #
            # if random.random() < exploration_prob:
            #     action = random.choice(list(eligible_actions))
            # else:
            #     action = q[eligible_actions].idxmax()


        elif rule == 'greedy':

            maxQ= max(q)
            #best_action = [i for i in range(len(q)) if q[i] == maxQ]

            if random.random() < epsilon:
                action = random.choice(list(eligible_actions))
            else:
                action = q[eligible_actions].idxmax()

            #actions.remove(action)
        else:
            #pr = np.zeros(len(eligible_actions))
            #cum_p1=0
            numer = list(map(lambda x: np.exp(x/T) ,q[eligible_actions]))
            denom = sum(numer)
            pr = numer/denom

            # for e,i in enumerate(eligible_actions):
            #     p1= np.exp(q[i]/T)
            #     pr[e] = p1
            #     cum_p1 = cum_p1 + p1
            # pr = pr/cum_p1

            #action = list(eligible_actions)[np.argmax(pr)]
            xk = list(eligible_actions)
            pk = pr
            #boltzmann_dist = stats.rv_discrete(name='boltzmann_dist', values=(xk, pr), seed=50)
            #action = boltzmann_dist.rvs()
            action = np.asscalar(np.random.choice(xk,1,p=pr))
        return action



    #@passbyvalue
    def updateDebris(self, action):

        #Update the rem_debris for the new state
        #copied_self = deepcopy(self)
        #new_rem_debris = copied_self.rem_debris
        new_rem_debris = self.rem_debris
        resource_usage = self.rem_debris[action]
        new_rem_debris[action] = 0

        return new_rem_debris, resource_usage

    def establishSupplyConnection(self, new_node, G_restored):
        # get the supply that the new node is connected to
        s = [i for i, val in enumerate(self.rem_supply) if val > 0]
        connected_sup = []  # Supplies with non-zero supply that are reachable from the new node
        for sup in s:
            if nx.has_path(G_restored, sup, new_node):
                connected_sup.append(sup)

        return connected_sup

    def realizeDemand(self, new_node, dist, connected_sup, G_collapsed, Cost,reachable_nodes):
        #random.seed(42)

        #Whether the new_node connects to a supply or not, it is REALIZED
        #Hence if it is a demand node first associate its realized value for demand
        if (self.rem_demand[new_node] != 0) & (new_node not in reachable_nodes):  # new_node is a demand node and not discovered before
            dem = dist.rvs()  # Generate a random demand value
            #dem = dist.mean().round(0)
            self.rem_demand[new_node] = dem  # Update it to the realized demand value
        else:
            dem=0
        #get the demand nodes, including the new_node if its a demand
        demand_nodes = [ind for ind,val in enumerate(self.rem_demand) if val!=0]
        supply_nodes = [ind for ind,val in enumerate(self.rem_supply) if val!=0]

        if  len(connected_sup)== 0:
            satisfied_demand = 0
            demand_vec = self.rem_demand
            supply_vec = self.rem_supply
        else:
            Cost, check = funcs2.updateCost(new_node, connected_sup, demand_nodes, G_collapsed, Cost) #establishing the connection between the supply and new node (not like an adjacency matrix just indicating if there is path)
            if check ==1:
                demand_vec, supply_vec, satisfied_demand = funcs2.supplyAllocation(Cost, self.rem_demand, self.rem_supply, demand_nodes, supply_nodes)
            else: #If nothing has changed - no need to make any calculations
                satisfied_demand = 0
                demand_vec = self.rem_demand
                supply_vec = self.rem_supply

        return demand_vec, supply_vec, satisfied_demand, dem


    def transferSupply(self, connected_supply):

        #Pick the supply node with lowest index that all supply that is going to be transfered
        sf = sorted(connected_supply)[0]
        tot = np.asarray(self.rem_supply)[connected_supply].sum()
        self.rem_supply[sf] = tot

        for i in connected_supply[1:]:
            self.rem_supply[i] = 0