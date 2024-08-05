from network_update import *
from topo import *
import gurobipy as gp
import json
import numpy as np
import time
import random

def transform_failure_set(failures):
    output = []
    for failure in failures:
        if type(failure) == int:
            output.append([failure])
        else:
            output.append(failure)
    return output

if __name__ == '__main__':
    
    topologies = []
    
    for i in range(6, 16):
        for j in range(10):
            topoName = "%d_%d" % (i, j)
            topologies.append(topoName)
    
    topologies += ['Quest', 'Internode', 'Dataxchange', 'Pern', 'Internetmci', 'Aconet', 'Niif', 'Netrail', 'HostwayInternational', 'Abilene', 'Noel', 'Heanet', 'Belnet2004', 'WideJpn', 'Cesnet200511', 'Cesnet200603', 'Pacificwave', 'BsonetEurope', 'GtsRomania', 'BtEurope', 'Globalcenter', 'Karen', 'Garr199904', 'Claranet', 'Marnet', 'Ernet', 'Renater2001', 'Highwinds', 'Fatman', 'Aarnet', 'Garr200404', 'Sprint', 'Latnet', 'Airtel', 'Iinet', 'Uninet', 'Nsfnet', 'Belnet2003', 'HiberniaUs', 'BtAsiaPac', 'Cesnet200706', 'Cesnet200304', 'Packetexchange', 'Fccn', 'Janetlense', 'KentmanAug2005', 'Navigata', 'Harnet', 'Garr199901', 'Easynet', 'Rhnet', 'Restena', 'Compuserve', 'GtsSlovakia', 'Garr200109', 'Sinet', 'Goodnet', 'Rediris', 'Agis', 'Geant2001', 'Gridnet', 'HurricaneElectric', 'Arpanet19719', 'Peer1', 'Ans', 'BtLatinAmerica', 'Renater2004', 'Rnp', 'Grnet', 'UniC', 'Ibm', 'Garr200112', 'Nextgen', 'Roedunet', 'Garr199905', 'Cesnet201006', 'Myren', 'HiberniaNireland', 'Eunetworks']
    
    topologies += ['RedBestel', 'PionierL3', 'HiberniaGlobal', 'Garr201105', 'RoedunetFibre', 'Garr201102', 'Garr201108', 'Belnet2008', 'Garr201109', 'Sanet', 'Oteglobe', 'IowaStatewideFiberMap', 'Garr201110', 'PionierL1', 'Arpanet19723', 'EliBackbone', 'Garr201111', 'Xeex', 'NetworkUsa', 'Palmetto', 'Intranetwork', 'Bics', 'Cwix', 'Geant2012', 'Ntt', 'Garr201103', 'Garr201010', 'Geant2010', 'Renater2008', 'Tinet', 'Bellcanada', 'CrlNetworkServices', 'Garr200902', 'Shentel', 'Iris', 'SwitchL3', 'Belnet2010', 'Janetbackbone', 'Bellsouth', 'Belnet2007', 'AttMpls', 'Dfn', 'Garr201008', 'Iij', 'Renater2010', 'Biznet', 'Intellifiber', 'Garr201001', 'Garr200908', 'Integra', 'Arnes', 'Garr201107', 'BeyondTheNetwork', 'Evolink', 'Surfnet', 'UsSignal', 'Garr201007', 'Belnet2006', 'Darkstrand', 'Garr201101', 'Garr201004', 'Tw', 'Switch', 'Cernet', 'Chinanet', 'Garr201012', 'Digex', 'Xspedius', 'Garr200912', 'Funet', 'Garr201201', 'Uunet', 'Ntelos', 'Canerie', 'Sunet', 'Globenet', 'Arpanet19728', 'Uninett2011', 'Esnet', 'Garr201005', 'Garr201112', 'BtNorthAmerica', 'Belnet2005', 'Columbus', 'Garr201003', 'Abvt', 'LambdaNet', 'Bandcon', 'Geant2009', 'Garr200909', 'AsnetAm', 'Belnet2009', 'Oxford', 'Uninett2010', 'Missouri', 'Renater2006',  'Garr201104', 'GtsPoland']
    

    for topo_name in topologies:
        infilePrefix = "./topo_info/"
        env = ReadTopo(infilePrefix, topo_name)
        nodeNum, linkNum, linkSet, demNum, demands, totalPathNum, pathSet, demRates, cMatrix, wMatrix, MAXWEIGHT = env.read_info()
        
        with gp.Env() as env:
            result_file = open("./results/%s.json" % (topo_name), "w")
            results = {}
            
            failure_set_file = open("./failure_set/%s.json" % (topo_name))

            max_failure_pr = float(failure_set_file.readline())
            link_capacities = json.loads(failure_set_file.readline())
            dem_rate = np.array(json.loads(failure_set_file.readline()))
            failure_set = transform_failure_set(json.loads(failure_set_file.readline()))
            select_failure_set = transform_failure_set(json.loads(failure_set_file.readline()))
            predict_sort_failure_set = transform_failure_set(json.loads(failure_set_file.readline()))
            critical_failure_set = transform_failure_set(json.loads(failure_set_file.readline()))
            
            for i in range(linkNum):
                linkSet[i][3] = link_capacities[i]
            
            print("failure_set:", len(failure_set), failure_set)
            print("selected failure set new:", select_failure_set)
            print('max_failure_pr:', max_failure_pr)

            select_num = len(select_failure_set)
            random_select_failure_set = random.sample(failure_set, select_num)

            objval, utilities, demand_utilities = mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, env=env) 
            print("objval:", objval)
            dem_rate = dem_rate / objval
            
        

            dem_rate = dem_rate / max_failure_pr * 1.25 
            results['max_failure_pr'] = max_failure_pr
            
            # original optimization problem
            
            start = time.time()
            objval, capa_aug = mcfsolver_networkupgrade(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, failure_set, env=env) 
            end = time.time()
            print("Full failure set network upgrade:", objval, end -start, capa_aug)
            results['Full'] = [objval, end-start, capa_aug]
            
            start = time.time()
            mlu, mlu_failure = mcfsolver_validation_ver2(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, failure_set, capa_aug, env=env) 
            end = time.time()
            print("Full failure set network validation for FERNresult:", mlu, mlu_failure, end -start)
            results['Full-validate'] = [mlu, end-start]
            

            
            start = time.time()
            objval, capa_aug = mcfsolver_networkupgrade(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, select_failure_set, phi=1, env=env) 
            end = time.time()
            print("Selected failure set network update:", objval, end -start, capa_aug)
            results['FERN'] = [objval, end-start, capa_aug]

            
            start = time.time()
            mlu, mlu_failure = mcfsolver_validation_ver2(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, failure_set, capa_aug, env=env) 
            end = time.time()
            print("Full failure set network validation for FERNresult:", mlu, mlu_failure, end -start)
            results['FERN-validate'] = [mlu, end-start]
            

            # random selected failure scenarios
            '''
            start = time.time()
            objval, capa_aug = mcfsolver_networkupgrade(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, random_select_failure_set, phi=1, env=env) 
            end = time.time()
            print("Randomly Selected failure set network update:", objval, end -start, capa_aug)
            '''

            # Ground-truth results
            
            start = time.time()
            objval, capa_aug = mcfsolver_networkupgrade(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, critical_failure_set, phi=1, env=env) 
            end = time.time()
            print("Ground-truth selected failure set network update:", objval, end -start, capa_aug)
            results['Ground-truth'] = [objval, end-start, capa_aug]

            
            start = time.time()
            mlu, mlu_failure = mcfsolver_validation_ver2(nodeNum, linkNum, demNum, demands, dem_rate, linkSet, wMatrix, MAXWEIGHT, failure_set, capa_aug, env=env) 
            end = time.time()
            print("Full failure set network validation for Ground-truth result:", objval, end -start)
            results['Ground-truth-validate'] = [mlu, end-start]
            
            print(json.dumps(results), file=result_file)
        

        
        
        
