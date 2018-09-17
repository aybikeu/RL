
from heapq import heappush, heappop
from itertools import count
import networkx as nx
import random
import funcs2

def SNEBC(G, demand, supply, weight):
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G

    supply_nodes = [i for i,p in enumerate(supply) if p>0]
    for s in supply_nodes:
        # single source shortest paths
        S, P, sigma, dist_label, sigma_dist = dijkstra_path(G, s, weight)
        # accumulation
        betweenness = accumulate(betweenness, S, P, sigma, s, sigma_dist, demand, supply, dist_label)
    # rescaling
    return betweenness


def dijkstra_path(G, s, weight='weight'):
    # modified from Eppstein
    S = []
    P = {}
    level = {}
    sigma_dist={}
    for v in G:
        P[v] = []
        #sigma_dist=[]
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0

    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []   # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        if v==s and next(c)==1:
            sigma[v] = 1.0
            #level[v]=0
        else:
            sigma[v] += sigma[pred]  # count paths
            sigma_dist[v,pred]=sigma[pred]

        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
                #level[w] = level[v]+1
            elif vw_dist == seen[w] and seen[w]!=0:  # handle equal paths
                if w in P[v]: #If w is a pred of v already - now you know v

                   sigma_dist[w, v] =  sigma[v] - sigma[w]
                   sigma[w] = sigma[v] #not max becuase you already added up w's sigma info to v by visiting w from v- you visited since its in pred info
                   # if P[v]==w: #w is the only pred
                   #     level[v]=level[w]
                   # level[w] = max(level[v],level[w])

                else:
                    sigma[w] += sigma[v]
                    sigma_dist[w,v]=sigma[v]
                    #level[w] = level[v]+1
                P[w].append(v)


    return S, P, sigma, seen, sigma_dist




def accumulate(betweenness, S, P, sigma, s, sigma_dist,demand, supply, dist_label):

    demand_nodes = [i for i, p in enumerate(demand) if p > 0]

    for d in demand_nodes:
        delta = dict.fromkeys(S, 0)
        beta = dict.fromkeys(S, 0)
        #sigma_dist[(d,d)] = 0
        #First get the subtree leading to the demand node
        S_demand = [d]
        S_tree = [d]
        while S_demand:
            p = S_demand.pop()
            pred_list = P[p]
            S_demand.extend(pred_list)
            S_tree.extend(pred_list)

        nodes_tree = set(S_tree)
        S_tree = [x for x in S if x in nodes_tree]
        dem = demand[d]
        sup = supply[s]

        if s in P[d]: #If s is demand's immediate predecessor, then:
            beta[d] = 1.0/ sigma[d]

        #Counting the number of paths each node appears from the demand node
        #start from demand node
        w = S_tree.pop()
        coef = sigma[w]
        while S_tree:
            for v in P[w]:
                delta[v] += sigma_dist[(w,v)]
            w = S_tree.pop()



        for w in nodes_tree:
            delta[w] = delta[w]/sigma[d]
            if w != s:
                betweenness[w] += (delta[w] + beta[w]) * (sup*dem) / (dist_label[d])


    return betweenness

def convert2edge (node_betw, edges):

    edge_betw = dict.fromkeys(edges, 0)

    for (i,v) in edges:
        edge_betw[(i,v)] = (node_betw[i] + node_betw[v])

    return edge_betw

def uncollapse (node_betw, G_collapsed):

    u = {}
    node_betw_uncol = []
    for n, d in G_collapsed.nodes(data=True):
        nodes = list(d['members'])
        for n2 in nodes:
            u[n2] = node_betw[n]

    for key in sorted(u.iterkeys()):
        node_betw_uncol.append(u[key])

    return node_betw_uncol

if __name__ == '__main__':

    demand = [0,0,3,3,0]
    supply = [5,0,0,0,5]
    EdgeListWeighted = [(0, 1, 1), (0, 2, 3), (1, 2, 3), (1, 3, 2),
                        (2, 3, 1), (2, 4, 3), (3, 4, 1)]
    G = nx.Graph()
    G.add_weighted_edges_from(EdgeListWeighted, weight='debris')

    btw = SNEBC(G, demand, supply,  weight='debris')
