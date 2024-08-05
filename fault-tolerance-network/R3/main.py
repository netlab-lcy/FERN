from r3 import *
from topo import *
import gurobipy as gp
import json
import numpy as np
import time
import random
import copy

def transform_failure_set_r3(failures):
    output = []
    for failure in failures:
        if type(failure) == int:
            #print(failure)
            output.append([failure * 2, failure * 2 + 1])
        else:
            failure_new = []
            for i in failure:
                failure_new.append(i * 2)
                failure_new.append(i * 2 + 1)
            output.append(failure_new)
    return output

def transform_failure_set(failures):
    output = []
    for failure in failures:
        if type(failure) == int:
            output.append([failure])
        else:
            output.append(failure)
    return output

if __name__ == '__main__':
    
    # set target topologies
    topologies = []
    for i in range(6, 16):
        for j in range(10):
            topoName = "%d_%d" % (i, j)
            topologies.append(topoName)
    
    topologies += ['Quest', 'Internode', 'Dataxchange', 'Pern', 'Internetmci', 'Aconet', 'Niif', 'Netrail', 'HostwayInternational', 'Abilene', 'Noel', 'Heanet', 'Belnet2004', 'WideJpn', 'Cesnet200511', 'Cesnet200603', 'Pacificwave', 'BsonetEurope', 'GtsRomania', 'BtEurope', 'Globalcenter', 'Karen', 'Garr199904', 'Claranet', 'Marnet', 'Ernet', 'Renater2001', 'Highwinds', 'Fatman', 'Aarnet', 'Garr200404', 'Sprint', 'Latnet', 'Airtel', 'Iinet', 'Uninet', 'Nsfnet', 'Belnet2003', 'HiberniaUs', 'BtAsiaPac', 'Cesnet200706', 'Cesnet200304', 'Packetexchange', 'Fccn', 'Janetlense', 'KentmanAug2005', 'Navigata', 'Harnet', 'Garr199901', 'Easynet', 'Rhnet', 'Restena', 'Compuserve', 'GtsSlovakia', 'Garr200109', 'Sinet', 'Goodnet', 'Rediris', 'Agis', 'Geant2001', 'Gridnet', 'HurricaneElectric', 'Arpanet19719', 'Peer1', 'Ans', 'BtLatinAmerica', 'Renater2004', 'Rnp', 'Grnet', 'UniC', 'Ibm', 'Garr200112', 'Nextgen', 'Roedunet', 'Garr199905', 'Cesnet201006', 'Myren', 'HiberniaNireland', 'Eunetworks']
    
    topologies += ['RedBestel', 'PionierL3', 'HiberniaGlobal', 'Garr201105', 'RoedunetFibre', 'Garr201102', 'Garr201108', 'Belnet2008', 'Garr201109', 'Sanet', 'Oteglobe', 'IowaStatewideFiberMap', 'Garr201110', 'PionierL1', 'Arpanet19723', 'EliBackbone', 'Garr201111', 'Xeex', 'NetworkUsa', 'Palmetto', 'Intranetwork', 'Bics', 'Cwix', 'Geant2012', 'Ntt', 'Garr201103', 'Garr201010', 'Geant2010', 'Renater2008', 'Tinet', 'Bellcanada', 'CrlNetworkServices', 'Garr200902', 'Shentel', 'Iris', 'SwitchL3', 'Belnet2010', 'Janetbackbone', 'Bellsouth', 'Belnet2007', 'AttMpls', 'Dfn', 'Garr201008', 'Iij', 'Renater2010', 'Biznet', 'Intellifiber', 'Garr201001', 'Garr200908', 'Integra', 'Arnes', 'Garr201107', 'BeyondTheNetwork', 'Evolink', 'Surfnet', 'UsSignal', 'Garr201007', 'Belnet2006', 'Darkstrand', 'Garr201101', 'Garr201004', 'Tw', 'Switch', 'Cernet', 'Chinanet', 'Garr201012', 'Digex', 'Xspedius', 'Garr200912', 'Funet', 'Garr201201', 'Uunet', 'Ntelos', 'Canerie', 'Sunet', 'Globenet', 'Arpanet19728', 'Uninett2011', 'Esnet', 'Garr201005', 'Garr201112', 'BtNorthAmerica', 'Belnet2005', 'Columbus', 'Garr201003', 'Abvt', 'LambdaNet', 'Bandcon', 'Geant2009', 'Garr200909', 'AsnetAm', 'Belnet2009', 'Oxford', 'Uninett2010', 'Missouri', 'Renater2006',  'Garr201104', 'GtsPoland']


    for topo_name in topologies:
        print("topo:", topo_name)
        infilePrefix = "./topo_info/"
        env = ReadTopo(infilePrefix, topo_name)
        nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()
        '''
        if nodeNum <= 70 or nodeNum > 80:
            continue
        '''

        with gp.Env() as env:
            result_file = open("./results/%s.json" % (topo_name), "w")
            results = {}

            # load TM, link_capacity, and failure set
            # we only use one case for each topology in this version 
            #dem_rate = np.array(demRates[0])
            with open("./failure_set/%s.json" % (topo_name)) as failure_set_file:
                max_failure_pr = float(failure_set_file.readline())
                link_capacities = json.loads(failure_set_file.readline())
                dem_rate = np.array(json.loads(failure_set_file.readline()))
                failure_set_normal = transform_failure_set(json.loads(failure_set_file.readline()))
                failure_set = transform_failure_set_r3(failure_set_normal)
                select_failure_set = transform_failure_set_r3(json.loads(failure_set_file.readline()))
                predict_sort_failure_set = transform_failure_set_r3(json.loads(failure_set_file.readline()))
                critical_failure_set = transform_failure_set_r3(json.loads(failure_set_file.readline()))
            for i in range(linkNum):
                linkSet[i][3] = link_capacities[i]
            print("link_capacities:", link_capacities)
            print("failure_set:", len(failure_set))
            print("selected failure set new:", select_failure_set)
            print('max_failure_pr:', max_failure_pr)

            select_num = int(len(predict_sort_failure_set) * 0.2)
            random_select_failure_set = random.sample(failure_set, select_num)

            objval, utilities, demand_utilities = mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, env=env) 
            print("objval:", objval)
            dem_rate = dem_rate / objval

            dem_rate = dem_rate / max_failure_pr * 0.4
            results['max_failure_pr'] = max_failure_pr
            results['link_capacity'] = link_capacities
            results['traffic_demand'] = dem_rate.tolist()


            
            # original optimization problem, as baseline in FERN 
            start = time.time()
            objval, utilities, demand_utilities, reroute_utilities, routes, reroutes = mcfsolver_origin(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, failure_set, env=env)
            end = time.time()
            print("Full failure set origin objval:", objval, 'time:', end - start)
            flag, failure, max_utility, congested_failures = validate(failure_set, routes, reroutes, utilities, nodeNum, linkNum, linkSet, demands, dem_rate)
            print("validate online:", flag, failure, max_utility)
            results['Full'] = [objval, end-start, objval, max_utility]
            
            
            # Final version of FERN
            start = time.time()
            print("calculate R3 with partially selected failure set: version 2.")
            objval, utilities, demand_utilities, reroute_utilities, routes, reroutes = mcfsolver_origin_aug(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, select_failure_set, env=env)
            end = time.time()
            print("objval:", objval, 'time:', end-start)

            flag, failure, max_utility, congested_failures_origin = validate_origin(failure_set, demand_utilities, reroute_utilities, demNum, linkNum, utilities)
            print("validate origin:", flag, failure, max_utility, congested_failures_origin)
            results['FERN-aug'] = [objval, end - start, max_utility]
            tmp_start = time.time()
            flag, failure, max_utility, congested_failures = validate(failure_set, routes, reroutes, utilities, nodeNum, linkNum, linkSet, demands, dem_rate)
            tmp_end = time.time()
            print("validate online:", flag, failure, max_utility)
            print("reaction time for each failure case:", (tmp_end-tmp_start)/len(failure_set))
            results['FERN-aug'].append(max_utility)
            
            iter_num = 0
            select_failure_set_tmp = copy.deepcopy(select_failure_set)
            while len(congested_failures_origin) > 0:
                print("select_failure_set_tmp:", select_failure_set_tmp)
                iter_num += 1
                tmp_flag = False 
                for failure in congested_failures_origin:
                    if not failure in select_failure_set_tmp:
                        select_failure_set_tmp.append(failure)
                        tmp_flag = True
                if not tmp_flag:
                    break
                
                print("calculate R3 with partially selected failure set: version 2. extra failures")
                objval, utilities, demand_utilities, reroute_utilities, routes, reroutes = mcfsolver_origin_aug(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, select_failure_set_tmp, env=env)

                flag, failure, max_utility, congested_failures_origin = validate_origin(failure_set, demand_utilities, reroute_utilities, demNum, linkNum, utilities)
                print("validate origin:", flag, failure, max_utility, congested_failures_origin)
                
                if nodeNum < 20:
                    if max_utility < 1.1:
                        break
                else:
                    if max_utility < 1.2:
                        break

            
            end = time.time()
            results['FERN2-aug'] = [iter_num, end - start - (tmp_end - tmp_start), max_utility]
            flag, failure, max_utility, congested_failures = validate(failure_set, routes, reroutes, utilities, nodeNum, linkNum, linkSet, demands, dem_rate)
            print("validate online:", flag, failure, max_utility)

            results['FERN2-aug'].append(max_utility)

            results['FERN2-aug-extra'] = copy.copy(results['FERN2-aug'])
            if max_utility > 1:
                tmp_flag = False 
                for failure in congested_failures:
                    if not failure in select_failure_set_tmp:
                        select_failure_set_tmp.append(failure)
                        tmp_flag = True
                if tmp_flag:
                    #start = time.time()
                    print("calculate R3 with partially selected failure set: version 2. extra failures")
                    objval, utilities, demand_utilities, reroute_utilities, routes, reroutes = mcfsolver_origin_aug(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, select_failure_set_tmp, env=env)
                    end = time.time()
                    flag, failure, max_utility, congested_failures = validate(failure_set, routes, reroutes, utilities, nodeNum, linkNum, linkSet, demands, dem_rate)
                    results['FERN2-aug-extra'][0] = iter_num + 1
                    results['FERN2-aug-extra'][1] = end - start - (tmp_end - tmp_start)
                    results['FERN2-aug-extra'][3] = max_utility
            
            print(json.dumps(results), file=result_file)
            result_file.close()
            


