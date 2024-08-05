from gurobipy import *
import copy


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

    # print("add conservation constraints")
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
Network upgrade problem in FERN
'''
def mcfsolver_networkupgrade(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, failureSet, phi=1.0, env=None):
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

    # reroute var
    capacityVarNum = linkNum
    capacityVarID = 0

    capacities = []
    for i in range(capacityVarNum):
        capacities.append(model.addVar(0, GRB.INFINITY, 0, GRB.INTEGER, "r"))



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
                    model.addConstr(sumout == inflow[k][j] * phi) 
                elif j == demands[k][1]:
                    model.addConstr(sumout == 0)
                    model.addConstr(sumin + inflow[k][j] * phi == 0)
                else:
                    model.addConstr(sumin + inflow[k][j] * phi == sumout)
    
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
            model.addConstr(sum1 <= flag * (linkSet[h][3] + capacities[h] * 256))
            model.addConstr(sum2 <= flag * (linkSet[h][3] + capacities[h] * 256))
        

    capacity_sum = 0
    for i in range(linkNum):
        capacity_sum += capacities[i]

    # Objective
    model.setObjective(capacity_sum, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    capa_aug = []
    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        for i in range(linkNum):
            capa_aug.append(capacities[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        
        
    return optVal, capa_aug


'''
original mcf optimizer for the link-backup routing problem
Equation (3) in the R3 paper
'''
def mcfsolver_validation(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, failureSet, capacity_aug=None, env=None):
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
    if capacity_aug == None:
        capacity_aug = [0] * linkNum

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
            model.addConstr(sum1 <= flag * (linkSet[h][3] + capacity_aug[h] * 256) * phi)
            model.addConstr(sum2 <= flag * (linkSet[h][3] + capacity_aug[h] * 256) * phi)
        

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

'''
based on mcf algorithm, cost much less memory
'''
def mcfsolver_validation_ver2(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, failureSet, capacity_aug=None, env=None):
    if capacity_aug == None:
        capacity_aug = [0] * linkNum
    
    linkSet = copy.deepcopy(linkSet)
    for i in range(linkNum):
        linkSet[i][3] += capacity_aug[i] * 256

    objval_origin, utilities, demand_utilities = mcfsolver(nodeNum, linkNum, demNum, demands, list(rates), linkSet, wMatrix, MAXWEIGHT, env=env)
    mlu = objval_origin
    mlu_failure = None
    for failure in failureSet:
        tmp = []
        for l in failure:
            tmp.append(linkSet[l][3])
            linkSet[l][3] = 0
        objval, utilities, demand_utilities = mcfsolver(nodeNum, linkNum, demNum, demands, list(rates), linkSet, wMatrix, MAXWEIGHT, env=env)
        if objval > mlu:
            mlu = objval
            mlu_failure = failure
            
        lind = 0
        for l in failure:
            linkSet[l][3] = tmp[lind]
            lind += 1 
        
    return mlu, mlu_failure