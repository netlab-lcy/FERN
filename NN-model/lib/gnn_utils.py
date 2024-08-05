import torch as th
import numpy as np
import copy
from torch_geometric.data import Data

def gnn_data_generator(node_num, link_num, linkSet, node_features_dim, edge_index, datas, mode, failure_type='multi', shuffle=False, sample=False, part_failure=False):
    if shuffle:
        np.random.shuffle(datas)
    ind = 0
    max_capa = 0
    for i in range(link_num):
        max_capa = max(max_capa, linkSet[i][3])
    
    global_edge_index = []
    for i in range(link_num):
        for j in range(link_num):
            global_edge_index.append([i, j])
    global_edge_index = th.tensor(global_edge_index).t().contiguous()
    
    # input state: [link_num (link); node_num*node_num (flow); routes_num (path); failure_num (failures)]
    for data in datas:
        max_util = max(data['utilities'])
        # for various link capacities
        if 'link_capacities' in data:
            max_capa = max(data['link_capacities'])
        sum_dmrate = sum(data['TM'])
        select_ind = []
        failure_set = []
        processed_edge_index = copy.deepcopy(edge_index)
        routes_num = 0
        routes_ratio = []
        
        # for flow node
        for i in range(node_num * (node_num - 1)):
            flow_routes = data['flow_routes'][i]
            routes_ratio += data['flow_split_ratios'][i]
            for route in flow_routes:
                edge = [node_num * node_num + link_num + routes_num, i + link_num]
                processed_edge_index.append(edge)
                edge = [i + link_num, node_num * node_num + link_num + routes_num]
                processed_edge_index.append(edge)
                for j in route:
                    edge = [j // 2, node_num * node_num + link_num + routes_num]
                    processed_edge_index.append(edge)
                    edge = [node_num * node_num + link_num + routes_num, j // 2]
                    processed_edge_index.append(edge)
                routes_num += 1

        y = [0] * (link_num + node_num * node_num + routes_num)
        mask = [0] * (link_num + node_num * node_num + routes_num)
        mask2 = [0] * (link_num + node_num * node_num + routes_num)
        prs = [0] * (routes_num + node_num * node_num + link_num)

        
        
        case_type_ind = [[] for i in range(3)] 
        # build up failure cases node  
        if failure_type == "multi":
            max_pr = np.array(data['performance_ratios_multi']).max()
            # for large topology P2 training, only part of failure scenarios have ground-truth labels
            if sample and part_failure:
                real_max_pr = max_pr
                tmp = []
                for i in range(link_num):
                    for j in range(i, link_num):
                        case_ind = i * link_num + j
                        if data['performance_ratios_multi'][i][j] >= 0.8 * max_pr:
                            tmp.append(data['weights2'][case_ind])
                tmp.sort()
                c_weight = tmp[int(len(tmp)*0.8)]

                max_pr = 1
                for i in range(link_num):
                    for j in range(i, link_num):
                        case_ind = i * link_num + j
                        if data['performance_ratios_multi'][i][j] < 0:
                            continue
                        
                        if data['weights2'][case_ind] < 0.9 and data['performance_ratios_multi'][i][j] < 0.8 * real_max_pr:
                            continue
                        if data['weights2'][case_ind] < c_weight and data['performance_ratios_multi'][i][j] >= 0.8 * real_max_pr:
                            continue
                        max_pr = max(max_pr, data['performance_ratios_multi'][i][j])
              
            for i in range(link_num):
                for j in range(i, link_num):
                    case_ind = i * link_num + j
                    if data['performance_ratios_multi'][i][j] < 0:
                        continue
                    
                    # Sample for failure impact regression model training
                    if mode == "normal" and sample:
                        temp_r = data['performance_ratios_multi'][i][j] / max_pr
                        if temp_r > 0.95:
                            sample_weight = 1.
                        elif temp_r > 0.8:
                            sample_weight = 0.5 + abs(data['weights'][case_ind]) / 0.4
                        else:
                            sample_weight = 0.01 + max(0, data['weights'][case_ind]) ** 2
                        if np.random.rand() > sample_weight:
                            continue
                        
                    # for large topo failure scenario sampling in P2 training
                    if sample and part_failure:
                        if data['weights2'][case_ind] < 0.9 and data['performance_ratios_multi'][i][j] < 0.8 * real_max_pr:
                            continue
                        if data['weights2'][case_ind] < c_weight and data['performance_ratios_multi'][i][j] >= 0.8 * real_max_pr:
                            continue
                    else:
                        pass
                    
                    select_ind.append(case_ind)
                    if i == j:
                        failure_set.append((i))
                    else:
                        failure_set.append((i, j))

                    if data['performance_ratios_multi'][i][j] / max_pr > 0.95:
                        case_type_ind[0].append(len(select_ind) - 1)
                    elif data['performance_ratios_multi'][i][j] / max_pr > 0.8:
                        case_type_ind[1].append(len(select_ind) - 1)
                    else:
                        case_type_ind[2].append(len(select_ind) - 1)
                    
                    edge = [i, routes_num + node_num * node_num + link_num - 1 + len(select_ind)]
                    processed_edge_index.append(edge)
                    if j != i:
                        edge = [j, routes_num + node_num * node_num + link_num - 1 + len(select_ind)]
                        processed_edge_index.append(edge)
                    
                    if mode == "normal":
                        target = data['performance_ratios_multi'][i][j]
                    elif mode == "classify":
                        if data['performance_ratios_multi'][i][j] / max_pr > 0.8:
                            target = 1.
                        else:
                            target = 0.
                    y.append(target)
                    prs.append(data['performance_ratios_multi'][i][j])
                    mask.append(1)
                    mask2.append(1)
        else:
            # We only apply triple failure for evaluation
            max_pr = np.array(data['performance_ratios_tripple']).max()
            for i in range(link_num):
                for j in range(i+1, link_num):
                    for k in range(j+1, link_num):
                        case_ind = i * link_num + j
                        if data['performance_ratios_tripple'][i][j][k] < 0:
                            continue

                        # for worst case strengthen
                        select_ind.append(case_ind)
                        failure_set.append((i, j, k))

                        if data['performance_ratios_tripple'][i][j][k] / max_pr > 0.95:
                            case_type_ind[0].append(len(select_ind) - 1)
                        elif data['performance_ratios_tripple'][i][j][k] / max_pr > 0.8:
                            case_type_ind[1].append(len(select_ind) - 1)
                        else:
                            case_type_ind[2].append(len(select_ind) - 1)
                        
                        edge = [i, routes_num + node_num * node_num + link_num - 1 + len(select_ind)]
                        processed_edge_index.append(edge)
                        edge = [j, routes_num + node_num * node_num + link_num - 1 + len(select_ind)]
                        processed_edge_index.append(edge)
                        edge = [k, routes_num + node_num * node_num + link_num - 1 + len(select_ind)]
                        processed_edge_index.append(edge)
                        
                        if mode == "normal":
                            target = data['performance_ratios_tripple'][i][j]
                        elif mode == "classify":
                            if data['performance_ratios_tripple'][i][j][k] / max_pr > 0.8:
                                target = 1.
                            else:
                                target = 0.
                        y.append(target)
                        prs.append(data['performance_ratios_tripple'][i][j][k])
                        mask.append(1)
                        mask2.append(1)
        
        for i in range(3):
            for j in case_type_ind[i]:
                mask[routes_num + node_num * node_num + link_num + j] /= len(case_type_ind[i]) 
                if i == 0 or i == 1:
                    mask2[routes_num + node_num * node_num + link_num + j] /= -(len(case_type_ind[0]) + len(case_type_ind[1]))
                else:
                    mask2[routes_num + node_num * node_num + link_num + j] /= len(select_ind) - len(case_type_ind[0]) - len(case_type_ind[1])
        

        processed_edge_index = th.tensor(processed_edge_index).t().contiguous()
        x = np.zeros((routes_num + node_num * node_num + link_num + len(select_ind), node_features_dim))

        # initial input state
        for k in range(link_num):
            y[k] = data['performance_ratios_single'][k] # unused 

            if 'link_capacities' in data:
                x[k][0] = data['link_capacities'][k] / max_capa
            else:
                x[k][0] = linkSet[k][3] / max_capa
            x[k][1] = data['utilities'][k * 2] / max_util
            x[k][2] = data['utilities'][k * 2 + 1] / max_util

            # extra
            x[k][3] = x[k][0] * data['utilities'][k * 2] / max_util
            x[k][4] = x[k][0] * data['utilities'][k * 2 + 1] / max_util
            x[k][5] = 1 # indicate node type

        # failure node
        for k in range(len(select_ind)):
            x[routes_num + node_num * node_num + link_num + k][6] = 1
        # flow node
        for k in range(node_num * (node_num - 1)):
            x[link_num + k][7] = 1
            x[link_num + k][8] = data['TM'][k] / sum_dmrate 
        # path node
        for k in range(routes_num):
            x[link_num + node_num * node_num + k][9] = 1
            x[link_num + node_num * node_num + k][10] = routes_ratio[k] 

        y = th.tensor(y, dtype=th.float32)
        input = Data(x = th.tensor(x), edge_index=processed_edge_index, y=y)
        input.mask = th.tensor(mask, dtype=th.float32)
        input.mask2 = th.tensor(mask2, dtype=th.float32)
        input.reverse_ind = [(case_ind, ind) for case_ind in select_ind]
        input.failure_set = failure_set
        input.prs = th.tensor(prs, dtype=th.float32)
        input.global_edge_index = global_edge_index
        input.link_capas = data['link_capacities']
        input.TM = data['TM']
        yield input

        ind += 1
        
        
        
    