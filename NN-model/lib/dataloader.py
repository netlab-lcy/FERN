import json
import random
import glob
import os
import re
import numpy as np
import torch as th
import copy

def load_data(data_dir, node_num, link_num, linkSet):
    data_file = open(data_dir, "r")
    datas = []

    Maps = {}
    for i in range(link_num):
        u = linkSet[i][0]
        v = linkSet[i][1]
        if u not in Maps:
            Maps[u] = []
        if v not in Maps:
            Maps[v] = []
        Maps[u].append(i * 2)
        Maps[v].append(i * 2 + 1)
    
    for line in data_file.readlines():
        data = json.loads(line)
        data['weights'] = np.ones(len(data['performance_ratios_single']) ** 2)
        np.random.seed(len(data['performance_ratios_single']) ** 2)
        data['weights2'] = np.random.random(len(data['performance_ratios_single']) ** 2) # large-scale P2 training
        
        # calculate the flow routes
        flow_routes = []
        flow_split_ratios = []
        demands = []
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    continue
                demands.append([src,dst])
        dem_num = len(demands)
        for i in range(dem_num):
            s = demands[i][0]
            t = demands[i][1]
            r = data['demand_utilities'][i]
            routes, split_ratio = getting_path(node_num, Maps, linkSet, r, s, t)
            flow_routes.append(routes)
            flow_split_ratios.append(split_ratio)
        
        del(data['demand_utilities']) # reduce memory usage
        
        data['flow_routes'] = flow_routes
        data['flow_split_ratios'] = flow_split_ratios
        datas.append(data)
        
        # for the memory limit, for OSPF data
        if node_num > 30:
            if len(datas) == 100:
                break
    return datas

def load_topo(data_dir):
    linkSet = []
    with open(data_dir, "r") as f:
        ind = 0
        for line in f.readlines():
            if ind == 0:
                node_num, link_num = list(map(int, line.split()))
            else:
                link = list(map(int, line.split()))
                link[0] -= 1
                link[1] -= 1
                linkSet.append(link)
            ind += 1
    return node_num, link_num, linkSet

def generate_edge_index(node_num, link_num, linkSet):
    node_adj_edge_set = [[] for i in range(node_num)]
    edge_index = []
    for i in range(link_num):
        for j in node_adj_edge_set[linkSet[i][0]]:
            edge = [i, j]
            edge_index.append(edge)
            edge = [j, i]
            edge_index.append(edge)
        node_adj_edge_set[linkSet[i][0]].append(i)
        for j in node_adj_edge_set[linkSet[i][1]]:
            edge = [i, j]
            edge_index.append(edge)
            edge = [j, i]
            edge_index.append(edge)
        node_adj_edge_set[linkSet[i][1]].append(i)
    
    return edge_index


def load_datas(data_dir):
    dataset_paths = glob.glob(os.path.join(data_dir, "*.data"))
    
    data_dict = {}
    for data_path in dataset_paths:
        topoName = re.findall(".*/(.+?).data", data_path)[0]
        print("loading dataset topoName:", topoName)
        node_num, link_num, linkSet = load_topo("./data/topology/%s_topo.txt" % (topoName))
        edge_index = generate_edge_index(node_num, link_num, linkSet)

        datas = load_data(data_path, node_num, link_num, linkSet)
        datas = np.array(datas)

        data_dict[topoName] = {
            'node_num': node_num,
            'link_num': link_num,
            'linkSet': linkSet,
            'edge_index': edge_index,
            'datas': datas,
        }
    return data_dict

def getting_path(nodeNum, Maps, linkSet, r, s, t):
    paths = []
    path_traffic = []

    tmp_r = copy.deepcopy(r)

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

        paths.append(path)
        path_traffic.append(utility_min)

        flag = [False] * nodeNum
        path = []

    path_traffic = (np.array(path_traffic) / sum(path_traffic)).tolist()
    return paths, path_traffic

def dfs(s, t, path, r, flag, Maps, linkSet):
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
    
