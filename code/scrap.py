from pulp import *

supply = [0,4]
demand = [2,3]

supply_amount = [0,0,0,0,4]
demand_amount = [0,0,3,2,0]

costs = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0]]

problem = LpProblem("Supply Allocation", LpMaximize)

paths = [(S,D) for S in supply for D in demand]
dvar = LpVariable.dicts("x", (supply,demand),0,None,LpInteger)

#Add to the objective function for the problem
problem += lpSum([dvar[s][t]*costs[s][t] for (s,t) in paths])

for s in supply:
    problem += lpSum([dvar[s][t] for t in demand if costs[s][t] > 0]) <= supply_amount[s]

for t in demand:
    problem += lpSum([dvar[s][t] for s in supply if costs[s][t] > 0 ]) <= demand_amount[t]

problem.writeLP('SupDem_Allocation.lp')
problem.solve()

for var in problem.variables():
    print "{} = {}".format(var.name, var.varValue)

