

from gurobipy import *
import copy
import numpy as np


'''
nodeNum: int
linkNum: int
demNum: int, indicate the number of flows, for a traffix you may have (n*(n-1)) flows
demands: list, flows' source and destination, each flow represented as a tuple
rates: list, show the flow demand (in traffic matrix)
linkSet: list, each comment represented as a tuple, i.e., (u, v, weight, capacity)
wMatrix and MAXWEIGHT: wMatrix[i][j] < MAXWEIGHT indicates that i,j is a legal link, otherwise, there no link between i,j
mode: 0: bi-directional link share the capacity; 1: bi-directional link do not share the capacity
background_utilities: link utilization ratio at each link from background traffic
env: gurobi environment, for multi-processing
'''
def mcfsolver(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, mode = 1, background_utilities=None, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]
    if background_utilities == None:
        background_utilities = [0.] * linkNum * 2


    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    # max link utilization constraints
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        if mode == 1:
            # when link failed, the link capacity is 0 and if and only if there is no flow going through the link, the constraints is achieved 
            model.addConstr(sum1 + background_utilities[h * 2] * linkSet[h][3] <= phi*linkSet[h][3])
            model.addConstr(sum2 + background_utilities[h * 2 + 1] * linkSet[h][3] <= phi*linkSet[h][3])
        else:
            model.addConstr(sum1 + sum2 + (background_utilities[h * 2] + background_utilities[h * 2 + 1]) * linkSet[h][3] <= phi*linkSet[h][3])

    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == inflow[k][j])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin + inflow[k][j] == 0)
            else:
                model.addConstr(sumin + inflow[k][j] == sumout)

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        # obtain representation vector, deprecated to save time
        utilities = [0.] * linkNum * 2
        demand_utilities = []
        for k in range(demNum):
            tmp = []
            for i in range(linkNum):
                util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                tmp.append(util1)  
                tmp.append(util2)
                utilities[i * 2] += util1 
                utilities[i * 2 + 1] += util2
            demand_utilities.append(tmp)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        utilities = []
        demand_utilities = []
    
    return optVal, utilities, demand_utilities


'''
Implementation of R3
'''
def mcfsolver_r3(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, F=2, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))

    # reroute var
    rerouteVarNum = linkNum * 2 * linkNum * 2
    rerouteVarID = 0
    rerouteMaps = {}
    for k in range(linkNum * 2):
        for i in range(linkNum):
            rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))] = rerouteVarID
            rerouteVarID += 1
            rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))] = rerouteVarID
            rerouteVarID += 1
    reroute = []
    for i in range(rerouteVarNum):
        reroute.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r"))

    # dual multipliers  \pi
    piVarNum = linkNum * 2 * linkNum * 2
    piVarID = 0
    piMaps = {}
    for i in range(linkNum * 2):
        for j in range(linkNum * 2):
            piMaps[(i, j)] = piVarID
            piVarID += 1
    pi = []
    for i in range(piVarNum):
        pi.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "pi"))
    
    # dual multipliers \lambda
    lamb = []
    for i in range(linkNum * 2):
        lamb.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "lamb"))

    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")


    # print("add conservation constraints")
    # for original flows
    
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == inflow[k][j])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin + inflow[k][j] == 0)
            else:
                model.addConstr(sumin + inflow[k][j] == sumout)
    
    # for reroute flows
    
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(linkNum * 2):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += reroute[rerouteMaps[(k, (i,j))]]
                    sumout += reroute[rerouteMaps[(k, (j,i))]]
            if k % 2 == 0:
                src = linkSet[k//2][0]
                dst = linkSet[k//2][1]
            else: 
                src = linkSet[k//2][1]
                dst = linkSet[k//2][0]
            if j == src:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == linkSet[k//2][3])
            elif j == dst:
                model.addConstr(sumout == 0)
                model.addConstr(sumin == linkSet[k//2][3])
            else:
                model.addConstr(sumin == sumout)
    

    # dual constraints
    
    for i in range(linkNum * 2):
        for j in range(linkNum * 2):
            if i % 2 == 0:
                model.addConstr(pi[piMaps[(i, j)]] + lamb[i]  >= reroute[rerouteMaps[j, (linkSet[i // 2][0], linkSet[i // 2][1])]])
            else:
                model.addConstr(pi[piMaps[(i, j)]] + lamb[i]  >= reroute[rerouteMaps[j, (linkSet[i // 2][1], linkSet[i // 2][0])]])
    
    # max link utilization constraints
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        # when link failed, the link capacity is 0 and if and only if there is no flow going through the link, the constraints is achieved 
        sum3 = 0
        sum4 = 0
        for k in range(linkNum * 2):
            sum3 += pi[piMaps[(h * 2, k)]]
            sum4 += pi[piMaps[(h * 2 + 1, k)]]
        model.addConstr(sum1 + sum3 + lamb[h * 2] * F <= phi*linkSet[h][3])
        model.addConstr(sum2 + sum4 + lamb[h * 2 + 1] * F  <= phi*linkSet[h][3])
        

    

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        # obtain representation vector, deprecated to save time
        utilities = [0.] * linkNum * 2
        demand_utilities = []
        for k in range(demNum):
            tmp = []
            for i in range(linkNum):
                util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                tmp.append(util1)  
                tmp.append(util2)
                utilities[i * 2] += util1 
                utilities[i * 2 + 1] += util2
            demand_utilities.append(tmp)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        utilities = []
        demand_utilities = []
    
    return optVal, utilities, demand_utilities

'''
original mcf optimizer for the link-backup routing problem
Equation (3) in the R3 paper
'''
def mcfsolver_origin(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, failureSet, r=None, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1
    
    flow = []
    if r == None:
        for i in range(flowVarNum):
            flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    else:
        for k in range(demNum):
            for i in range(linkNum * 2):
                flow.append(r[k][i] * linkSet[i//2][3])

    # reroute var
    rerouteVarNum = linkNum * 2 * linkNum * 2
    rerouteVarID = 0
    rerouteMaps = {}
    for k in range(linkNum * 2):
        for i in range(linkNum):
            rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))] = rerouteVarID
            rerouteVarID += 1
            rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))] = rerouteVarID
            rerouteVarID += 1
    reroute = []
    for i in range(rerouteVarNum):
        reroute.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r"))

    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")


    # print("add conservation constraints")
    # for original flows
    if r == None:
        sumpass = 0
        sumin = 0
        sumout = 0
        for k in range(demNum):
            for j in range(nodeNum):
                sumin = 0
                sumout = 0
                for i in range(nodeNum):
                    if wMatrix[i][j] < MAXWEIGHT and i != j:
                        sumin += flow[Maps[(k,(i,j))]]
                        sumout += flow[Maps[(k,(j,i))]]
                if j == demands[k][0]:
                    model.addConstr(sumin == 0)
                    model.addConstr(sumout == inflow[k][j])
                elif j == demands[k][1]:
                    model.addConstr(sumout == 0)
                    model.addConstr(sumin + inflow[k][j] == 0)
                else:
                    model.addConstr(sumin + inflow[k][j] == sumout)
    
    # for reroute flows
    
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(linkNum * 2):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += reroute[rerouteMaps[(k, (i,j))]]
                    sumout += reroute[rerouteMaps[(k, (j,i))]]
            if k % 2 == 0:
                src = linkSet[k//2][0]
                dst = linkSet[k//2][1]
            else: 
                src = linkSet[k//2][1]
                dst = linkSet[k//2][0]
            if j == src:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == linkSet[k//2][3])
            elif j == dst:
                model.addConstr(sumout == 0)
                model.addConstr(sumin == linkSet[k//2][3])
            else:
                model.addConstr(sumin == sumout)
    
    # max link utilization constraints
    for failure in failureSet:
        for h in range(linkNum):
            i = linkSet[h][0]
            j = linkSet[h][1]
            sum1 = 0
            sum2 = 0
            for k in range(demNum):
                sum1 += flow[Maps[(k,(i,j))]]
                sum2 += flow[Maps[(k,(j,i))]]
            
            sum3 = 0
            sum4 = 0
            for k in failure:
                sum3 += reroute[rerouteMaps[(k, (i, j))]]
                sum4 += reroute[rerouteMaps[(k, (j, i))]]
            model.addConstr(sum1 + sum3 <= phi*linkSet[h][3])
            model.addConstr(sum2 + sum4  <= phi*linkSet[h][3])
        

    

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal

        # obtain representation vector, deprecated to save time
        
        utilities = [0.] * linkNum * 2
        demand_utilities = []
        routes = []
    
        for k in range(demNum):
            tmp = []
            route = []
            for i in range(linkNum):
                if r == None:
                    util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                    util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                else:
                    util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]] / (linkSet[i][3] + 1e-5)
                    util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]] / (linkSet[i][3] + 1e-5)
                tmp.append(util1)  
                tmp.append(util2)
                route.append(util1 * linkSet[i][3] / rates[k])
                route.append(util2 * linkSet[i][3] / rates[k])
                utilities[i * 2] += util1 
                utilities[i * 2 + 1] += util2
            demand_utilities.append(tmp)
            routes.append(route)
        reroute_utilities = []
        reroutes = []
        for k in range(linkNum * 2):
            reroute_utility = []
            tmp_reroute = []
            for i in range(linkNum):
                util1 = reroute[rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = reroute[rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                reroute_utility.append(util1)  
                reroute_utility.append(util2)
                tmp_reroute.append(util1 * linkSet[i][3] / linkSet[k//2][3])
                tmp_reroute.append(util2 * linkSet[i][3] / linkSet[k//2][3])
                
            reroute_utilities.append(reroute_utility)
            reroutes.append(tmp_reroute)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        utilities = []
        demand_utilities = []
        reroute_utilities = []
        routes = []
        reroutes = []
    
    return optVal, utilities, demand_utilities, reroute_utilities, routes, reroutes

def mcfsolver_origin_aug(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, failureSet, r=None, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1
    
    flow = []
    if r == None:
        for i in range(flowVarNum):
            flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    else:
        for k in range(demNum):
            for i in range(linkNum * 2):
                flow.append(r[k][i] * linkSet[i//2][3])

    # reroute var
    rerouteVarNum = linkNum * 2 * linkNum * 2
    rerouteVarID = 0
    rerouteMaps = {}
    for k in range(linkNum * 2):
        for i in range(linkNum):
            rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))] = rerouteVarID
            rerouteVarID += 1
            rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))] = rerouteVarID
            rerouteVarID += 1
    reroute = []
    for i in range(rerouteVarNum):
        reroute.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r"))

    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")
    mus = []
    for i in range(linkNum):
        mus.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "mu"))

    # print("add conservation constraints")
    # for original flows
    if r == None:
        sumpass = 0
        sumin = 0
        sumout = 0
        for k in range(demNum):
            for j in range(nodeNum):
                sumin = 0
                sumout = 0
                for i in range(nodeNum):
                    if wMatrix[i][j] < MAXWEIGHT and i != j:
                        sumin += flow[Maps[(k,(i,j))]]
                        sumout += flow[Maps[(k,(j,i))]]
                if j == demands[k][0]:
                    model.addConstr(sumin == 0)
                    model.addConstr(sumout == inflow[k][j])
                elif j == demands[k][1]:
                    model.addConstr(sumout == 0)
                    model.addConstr(sumin + inflow[k][j] == 0)
                else:
                    model.addConstr(sumin + inflow[k][j] == sumout)
    
    # for reroute flows
    
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(linkNum * 2):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += reroute[rerouteMaps[(k, (i,j))]]
                    sumout += reroute[rerouteMaps[(k, (j,i))]]
            if k % 2 == 0:
                src = linkSet[k//2][0]
                dst = linkSet[k//2][1]
            else: 
                src = linkSet[k//2][1]
                dst = linkSet[k//2][0]
            if j == src:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == linkSet[k//2][3])
            elif j == dst:
                model.addConstr(sumout == 0)
                model.addConstr(sumin == linkSet[k//2][3])
            else:
                model.addConstr(sumin == sumout)
    
    # max link utilization constraints
    for failure in failureSet:
        for h in range(linkNum):
            i = linkSet[h][0]
            j = linkSet[h][1]
            sum1 = 0
            sum2 = 0
            for k in range(demNum):
                sum1 += flow[Maps[(k,(i,j))]]
                sum2 += flow[Maps[(k,(j,i))]]
            
            sum3 = 0
            sum4 = 0
            for k in failure:
                sum3 += reroute[rerouteMaps[(k, (i, j))]]
                sum4 += reroute[rerouteMaps[(k, (j, i))]]
            model.addConstr(sum1 + sum3 <= phi*linkSet[h][3])
            model.addConstr(sum2 + sum4  <= phi*linkSet[h][3])
    
    for link in range(linkNum):
        failure = [link * 2, link * 2 + 1]
        for h in range(linkNum):
            i = linkSet[h][0]
            j = linkSet[h][1]
            sum1 = 0
            sum2 = 0
            for k in range(demNum):
                sum1 += flow[Maps[(k,(i,j))]]
                sum2 += flow[Maps[(k,(j,i))]]
            
            sum3 = 0
            sum4 = 0
            for k in failure:
                sum3 += reroute[rerouteMaps[(k, (i, j))]]
                sum4 += reroute[rerouteMaps[(k, (j, i))]]
            model.addConstr(sum1 + sum3 <= mus[link]*linkSet[h][3])
            model.addConstr(sum2 + sum4  <= mus[link]*linkSet[h][3])

    # for testing: congestion constraint     
    model.addConstr(phi  <= 1.)
    

    # Objective
    model.setObjective(phi+sum(mus)/linkNum, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal

        # obtain representation vector, deprecated to save time
        utilities = [0.] * linkNum * 2
        demand_utilities = []
        routes = []
    
        for k in range(demNum):
            tmp = []
            route = []
            for i in range(linkNum):
                if r == None:
                    util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                    util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                else:
                    util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]] / (linkSet[i][3] + 1e-5)
                    util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]] / (linkSet[i][3] + 1e-5)
                tmp.append(util1)  
                tmp.append(util2)
                route.append(util1 * linkSet[i][3] / rates[k])
                route.append(util2 * linkSet[i][3] / rates[k])
                utilities[i * 2] += util1 
                utilities[i * 2 + 1] += util2
            demand_utilities.append(tmp)
            routes.append(route)
        reroute_utilities = []
        reroutes = []
        for k in range(linkNum * 2):
            reroute_utility = []
            tmp_reroute = []
            for i in range(linkNum):
                util1 = reroute[rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = reroute[rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                reroute_utility.append(util1)  
                reroute_utility.append(util2)
                tmp_reroute.append(util1 * linkSet[i][3] / linkSet[k//2][3])
                tmp_reroute.append(util2 * linkSet[i][3] / linkSet[k//2][3])
                
            reroute_utilities.append(reroute_utility)
            reroutes.append(tmp_reroute)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        utilities = []
        demand_utilities = []
        reroute_utilities = []
        routes = []
        reroutes = []
    
    return optVal, utilities, demand_utilities, reroute_utilities, routes, reroutes



'''
augmented mcf optimizer for the link-backup routing problem
for single link failure it optimized the link backup routing independently
given the original routing r(in R3)
'''
def mcfsolver_aug(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, r, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    link_list = [[] for i in range(nodeNum)]
    Maps = {}
    for i in range(linkNum):
        link_list[linkSet[i][0]].append(linkSet[i][1])
        link_list[linkSet[i][1]].append(linkSet[i][0])
        Maps[(linkSet[i][0], linkSet[i][1])] = i * 2
        Maps[(linkSet[i][1], linkSet[i][0])] = i * 2 + 1

    # calculate link associated links(reachable through 2-degree nodes)
    link_associated = [[] for i in range(linkNum)]
    for i in range(linkNum):
        flag = [False for i in range(nodeNum)]
        s = linkSet[i][0]
        t = linkSet[i][1]
        flag[s] = True
        flag[t] = True
        cur_p = t
        while len(link_list[cur_p]) == 2:
            for j in link_list[cur_p]:
                if not flag[j]:
                    link_associated[i].append(Maps[(cur_p, j)] // 2)
                    cur_p = j
                    break
            if not flag[cur_p]:
                flag[cur_p] = True
            else:
                break
        cur_p = s
        while len(link_list[cur_p]) == 2:
            for j in link_list[cur_p]:
                if not flag[j]:
                    link_associated[i].append(Maps[(cur_p, j)] // 2)
                    cur_p = j
                    break
            if not flag[cur_p]:
                flag[cur_p] = True
            else:
                break
    
    # reroute var
    rerouteVarNum = linkNum * 2 * linkNum * 2
    rerouteVarID = 0
    rerouteMaps = {}
    for k in range(linkNum * 2):
        for i in range(linkNum):
            rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))] = rerouteVarID
            rerouteVarID += 1
            rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))] = rerouteVarID
            rerouteVarID += 1
    reroute = []
    for i in range(rerouteVarNum):
        reroute.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r"))

    

    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")
    mus = []
    for i in range(linkNum * 2):
        mus.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "mu"))


    # print("add conservation constraints")
    # for reroute flows
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(linkNum * 2):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += reroute[rerouteMaps[(k, (i,j))]]
                    sumout += reroute[rerouteMaps[(k, (j,i))]]
            if k % 2 == 0:
                src = linkSet[k//2][0]
                dst = linkSet[k//2][1]
            else: 
                src = linkSet[k//2][1]
                dst = linkSet[k//2][0]
            if j == src:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == linkSet[k//2][3])
            elif j == dst:
                model.addConstr(sumout == 0)
                model.addConstr(sumin == linkSet[k//2][3])
            else:
                model.addConstr(sumin == sumout)
    
    # max link utilization constraints
    for failure in range(linkNum * 2):
        for h in range(linkNum):
            i = linkSet[h][0]
            j = linkSet[h][1]
            sum1 = 0
            sum2 = 0
            for k in range(demNum):
                sum1 += r[k][Maps[(i, j)]] * linkSet[h][3]
                sum2 += r[k][Maps[(j, i)]] * linkSet[h][3]
            
            sum3 = 0
            sum4 = 0
            for k in [failure]:
                sum3 += reroute[rerouteMaps[(k, (i, j))]]
                sum4 += reroute[rerouteMaps[(k, (j, i))]]

            # ensure the flow better load balanced
            # for the condition that degree = 1 node exist
            if h in link_associated[failure // 2]:
                delta = 10.
            else:
                delta = 1.
            
            model.addConstr(sum1 + sum3 <= mus[failure]*linkSet[h][3] * delta)
            model.addConstr(sum2 + sum4  <= mus[failure]*linkSet[h][3] * delta)
        

    

    # Objective
    model.setObjective(sum(mus), GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal

        # obtain representation vector, deprecated to save time
        reroute_utilities = []
        for k in range(linkNum * 2):
            reroute_utility = []
            for i in range(linkNum):
                util1 = reroute[rerouteMaps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = reroute[rerouteMaps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                reroute_utility.append(util1)  
                reroute_utility.append(util2)
                
            reroute_utilities.append(reroute_utility)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        demand_utilities = []
        reroute_utilities = []
    
    return optVal, r, reroute_utilities




'''
Robust validation mcf model
'''
def mcfsolver_validation(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, failureSet, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]
    failureNum = len(failureSet)

    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * failureNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for j in range(failureNum):
        for k in range(demNum):
            for i in range(linkNum):
                Maps[(j, k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
                flowVarID += 1
                Maps[(j, k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
                flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))


    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    # print("add conservation constraints")
    # for flow conservation constraints
    sumpass = 0
    sumin = 0
    sumout = 0
    for find in range(failureNum):
        for k in range(demNum):
            for j in range(nodeNum):
                sumin = 0
                sumout = 0
                for i in range(nodeNum):
                    if wMatrix[i][j] < MAXWEIGHT and i != j:
                        sumin += flow[Maps[(find, k,(i,j))]]
                        sumout += flow[Maps[(find, k,(j,i))]]
                if j == demands[k][0]:
                    model.addConstr(sumin == 0)
                    model.addConstr(sumout == inflow[k][j])
                elif j == demands[k][1]:
                    model.addConstr(sumout == 0)
                    model.addConstr(sumin + inflow[k][j] == 0)
                else:
                    model.addConstr(sumin + inflow[k][j] == sumout)
    
    # congestion free constraints
    for find in range(failureNum):
        for h in range(linkNum):
            i = linkSet[h][0]
            j = linkSet[h][1]
            sum1 = 0
            sum2 = 0
            for k in range(demNum):
                sum1 += flow[Maps[(find, k,(i,j))]]
                sum2 += flow[Maps[(find, k,(j,i))]]
            if h in failureSet[find]:
                flag = 0
            else:
                flag = 1
            model.addConstr(sum1 <= flag * linkSet[h][3] * phi)
            model.addConstr(sum2 <= flag * linkSet[h][3] * phi)
        

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
       
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        
    return optVal


def dfs(s, t, path, r, flag, Maps, linkSet):
    #print("path", path, "flag:", flag)
    flag[s] = True 
    if s == t: 
        return True
    for l in Maps[s]:
        if l % 2 == 0:
            v = linkSet[l // 2][1]
        else:
            v = linkSet[l // 2][0]
        if not flag[v] and r[l] > 1e-6:
            path.append(l)
            if dfs(v, t, path, r, flag, Maps, linkSet):
                return True 
            else:
                path.pop()

    flag[s] = False
    return False

'''
factor: normalization factor
'''
def eliminate_loop(nodeNum, linkSet, r, s, t, factor=1.):
    tmp_r = copy.deepcopy(r)
    linkNum = len(r) // 2
    Maps = {}
    for i in range(linkNum):
        u = linkSet[i][0]
        v = linkSet[i][1]
        if u not in Maps:
            Maps[u] = []
        if v not in Maps:
            Maps[v] = []
        Maps[u].append(i * 2)
        Maps[v].append(i * 2 + 1)

    path = []
    flag = [False] * nodeNum
    utility_sum = 0
    while dfs(s, t, path, tmp_r, flag, Maps, linkSet):
        utility_min = 1e6
        for i in path:
            utility_min = min(utility_min, tmp_r[i])
        utility_sum += utility_min
        for i in path:
            tmp_r[i] -= utility_min
        #print("path", path, "utility_min", utility_min)
        flag = [False] * nodeNum
        path = []
    #print("utility_sum:", utility_sum)
    r_noloop = ((np.array(r) - np.array(tmp_r)) / utility_sum  * factor).tolist() # with re-normalization
    #print("tmp_r", tmp_r, "r_no_loop", r_noloop)
    return r_noloop
    

'''
    raw validate algorithm based on R3 offline algorithm
'''
def validate_origin(failureSet, r, p, demNum, linkNum, utilities):
    congested_failures = []
    congested_failures_critical = []
    max_utility_sum = 0.
    max_failure = None
    flag = True
    for failure in failureSet:
        failure_mlu = 0.
        for i in range(linkNum * 2):
            utility_sum = 0.
            for j in range(demNum):
                utility_sum += r[j][i]
            for j in failure:
                utility_sum += p[j][i]
            failure_mlu = max(utility_sum, failure_mlu)
            if utility_sum > max_utility_sum:
                max_utility_sum = utility_sum
                max_failure = failure
            if utility_sum > 1 + 1e-3:
                flag = False
        if failure_mlu > 1 + 1e-3:
            congested_failures.append((failure, failure_mlu))

    for failure in congested_failures:
            
            if failure[1] > max(1., 0.8 * max_utility_sum):
                congested_failures_critical.append(failure[0])
    return flag, max_failure, max_utility_sum, congested_failures_critical



'''
validation algorithm based on R3 online reconfiguration algorithm
** note that r and p is proportion of the flow assigned at each edge
** note that the sum of p[i] = 1 with the assumption 
** this algorithm calculate the upper bound of failure impact
'''
def validate(failureSet, r, p, utilities, nodeNum, linkNum, linkSet, demands, dem_rate):
    congested_failures = []
    max_utility_sum = 0.
    max_failure = None
    flag = True
    for failure in failureSet:
        failure_flag = True
        tmp_utilities = copy.deepcopy(utilities)
        tmp_p = copy.deepcopy(p)
        tmp_r = copy.deepcopy(r)
        ind = 0
        for i in failure:
            if not failure_flag:
                break
            ind += 1
            if i % 2 == 0:
                s = linkSet[i // 2][0]
                t = linkSet[i // 2][1]
            else:
                s = linkSet[i // 2][1]
                t = linkSet[i // 2][0]
            for j in failure[:ind-1]:
                tmp_p[i][j] = 0
            

            if 1 - tmp_p[i][i] < 1e-6:
                if tmp_utilities[i] > 1e-6:
                    failure_flag = False
                    for j in failure[ind:]:
                        if tmp_p[j][i] > 1e-6:
                            failure_flag = False
                continue

            ksi = 1 / (1 - tmp_p[i][i])
            tmp_p[i][i] = 0
            
            for j in range(linkNum * 2):
                tmp_utilities[j] += tmp_utilities[i] * linkSet[i//2][3] * tmp_p[i][j] * ksi / linkSet[j//2][3]
            
            for k in range(len(tmp_r)):
                for j in range(linkNum * 2):
                    tmp_r[k][j] += tmp_r[k][i] * tmp_p[i][j] * ksi
            for k in failure[ind:]:
                for j in range(linkNum * 2):
                    tmp_p[k][j] += tmp_p[k][i] * tmp_p[i][j] * ksi
            #print("tmp_utilities", tmp_utilities)
        if not failure_flag:
            max_failure = failure 
            max_utility_sum = 1e6
            flag = False 
            congested_failures.append(failure)
        else:
            for i in failure:
                tmp_utilities[i] = 0.   
                for k in range(len(tmp_r)):
                    tmp_r[k][i] = 0.
            
            tmp_utilities = np.array(tmp_utilities)
            if np.max(tmp_utilities) > 1:
                flag = False
                congested_failures.append(failure)
            if np.max(tmp_utilities) > max_utility_sum:
                max_utility_sum = np.max(tmp_utilities)
                max_failure = failure
    return flag, max_failure, max_utility_sum, congested_failures



