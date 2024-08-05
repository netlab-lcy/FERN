from lib.dataloader import load_datas
from config.arguments import get_arg
from lib.nnmodel import ResGAT
from lib.gnn_utils import gnn_data_generator
import numpy as np
import torch as th
import torch.nn as nn
from torch_geometric.data import  DataLoader

import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, auc

from scipy import stats
import json
import time



def calc_gap_accuracy(predict, target, gap = 0.1, ratio_lowerbound=0.05, ratio_upperbound=0.1):
    flag = False
    max_predict_val = -1e6
    val_num = len(predict)

    predict = np.array(predict)
    target = np.array(target)
    max_target_val = target.max()
    predict_sort_ind = (-predict).argsort()
    selected_case_num = val_num
    select_failure_set = list(predict_sort_ind)
    
    worst_case_num = 0
    worst_case_num_predicted = 0.

    for i in target:
        if i > max_target_val * 0.8:
            worst_case_num += 1

    for i in range(val_num):
        #print(i, target[predict_sort_ind[i]], max_target_val)
        max_predict_val = max(max_predict_val, target[predict_sort_ind[i]])
        if target[predict_sort_ind[i]] > max_target_val - 1e-3:
            flag = True

        if target[predict_sort_ind[i]] > 0.8 * max_target_val:
            worst_case_num_predicted += 1

        if (i+1) / val_num > ratio_lowerbound and ((predict[predict_sort_ind[0]] - predict[predict_sort_ind[i]]) / predict[predict_sort_ind[0]] > gap or (i + 1) / val_num > ratio_upperbound):
            selected_case_num = i + 1
            select_failure_set = predict_sort_ind.tolist()[:i+1]
            break
    ae = max_target_val - max_predict_val
    re = ae / max_target_val
    return flag, worst_case_num_predicted/worst_case_num, ae, re, selected_case_num, val_num, select_failure_set



if __name__ == "__main__":
    # arguments
    args = get_arg()
    device = 'cuda' if args.use_cuda and th.cuda.is_available() else 'cpu'
    print("device:", device)
    mode = args.mode 
    log_dir = args.log_dir
    failure_type = args.failure_type

    # model parameter
    node_features_dim = args.node_features_dim
    hidden_units = args.hidden_units 
    local_heads = args.global_heads
    global_heads = args.local_heads
    output_units = 1

    model = ResGAT(node_features_dim, output_units, hidden_units, local_heads, global_heads)
    model.load_state_dict(th.load("./models/%s/model.th" % (log_dir)))
    model.to(device)
    model.eval()
    
    print("loading dataset...")
    eval_data_dir = args.eval_data_dir
    eval_data_dict = load_datas("./data/dataset/"+eval_data_dir)
    
    if mode == "classify":
        activator = nn.Sigmoid()
    
    # evaluation
    re_all = []
    re_critical = []
    eval_log_file = open("./log/%s/%s.log" % (log_dir, eval_data_dir), "w")

    # for classify
    classify_y = []
    classify_label = []

    for topo_name in eval_data_dict:
        print("Evaluate Topo Name:", topo_name)
        print("Evaluate Topo Name:", topo_name, file=eval_log_file)
        data_list = []
        eval_data = eval_data_dict[topo_name]
        eval_data_generator = gnn_data_generator(eval_data['node_num'], eval_data['link_num'], eval_data['linkSet'], node_features_dim, eval_data['edge_index'], eval_data['datas'], mode, failure_type=failure_type, shuffle=False, sample=False) 
        for data in eval_data_generator:
            data_list.append(data)
        eval_loader = DataLoader(data_list, batch_size=1)
        
        for batch in eval_loader:
            batch.to(device)
            with th.no_grad():
                start = time.time()
                y = model(batch)
                end = time.time()
                print("time overhead:", (end - start))
                
                if mode == 'classify':
                    y = activator(y)
                    ret = y.cpu().numpy().flatten().tolist()
                    mask =  (batch.mask / (batch.mask + 1e-9)).cpu().numpy().flatten().tolist()
                    label_classify = batch.y.cpu().numpy().flatten().tolist()
                    label_pr = batch.prs.cpu().numpy().flatten().tolist()
                    predict = []
                    target = []
                    target_prs = []
                    for i in range(len(ret)):
                        if mask[i] != 0:
                            predict.append(ret[i])
                            target.append(label_classify[i])
                            target_prs.append(label_pr[i])

                            classify_y.append(ret[i])
                            classify_label.append(label_classify[i])

                    compare_list = []
                    sorted_index = (-np.array(target_prs)).argsort()
                    selected_failure_num = 0
                    p_case_num = 0
                    acc_p_case_num = 0
                    for i in sorted_index:
                        compare_list.append((predict[i], target[i], target_prs[i]))

                    print("compare_list:", compare_list, file=eval_log_file)
                            
                else:
                    ret = ((y.detach().squeeze(-1) - batch.y.detach()) / (1e-3 + batch.y.detach())).cpu().numpy().flatten().tolist()
                    mask =  (batch.mask / (batch.mask + 1e-9)).cpu().numpy().flatten().tolist()
                    label = batch.y.cpu().numpy().flatten().tolist()
                    predict = []
                    target = []
                    
                    max_val = max(label)
                    for i in range(len(ret)):
                        if mask[i] != 0:
                            predict.append(y.squeeze(-1).detach()[i].item())
                            target.append(label[i])
                            re_all.append(ret[i])
                            if label[i] > max_val * 0.8:
                                re_critical.append(ret[i])
        
        
                    compare_list = []
                    sorted_index = (-np.array(target)).argsort()
                    for i in sorted_index:
                        compare_list.append((predict[i], target[i], predict[i] - target[i]))
                    print("compare_list:", compare_list, file=eval_log_file)
                    print("spearman correlation coefficient:", stats.spearmanr(predict, target), file=eval_log_file)
    
    if mode == "normal":
        print("critical mre:", np.abs(np.array(re_critical)).mean(), file=eval_log_file)
        print("mre:",  np.abs(np.array(re_all)).mean(), file=eval_log_file)
    else:
        fpr, tpr, thresholds = roc_curve(classify_label, classify_y)
        roc_auc = auc(fpr,tpr)
        print("roc-auc:", roc_auc, file=eval_log_file)
    
    if mode == "normal":
        re_all_str = json.dumps(re_all)
        log_file = open('./log/%s/%s-re.json' % (log_dir, eval_data_dir), "w")
        print(re_all_str, file=log_file)

        re_critical_str = json.dumps(re_critical)
        log_file = open('./log/%s/%s-re-critical.json' % (log_dir, eval_data_dir), "w")
        print(re_critical_str, file=log_file)

    elif mode == 'classify':
        results_data = {'y': classify_label, 'prob': classify_y}
        results_str = json.dumps(results_data)
        log_file = open('./log/%s/%s-classify-ret.json' % (log_dir, eval_data_dir), "w")
        print(results_str, file=log_file)
    
