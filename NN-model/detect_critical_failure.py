from lib.dataloader import load_datas
from config.arguments import get_arg
from lib.nnmodel import ResGAT
from lib.gnn_utils import gnn_data_generator
from lib.utils import cleanup_dir, smooth
import numpy as np
import torch as th
import torch.nn as nn
from torch_geometric.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time





def calc_gap_accuracy(predict_pr, target_pr, predict_classify, threshold_classify, gap = 0.2, ratio_lowerbound=0.05, ratio_upperbound=0.1):
    flag = False
    max_predict_val = -1e6
    val_num = len(predict_pr)

    predict_pr = np.array(predict_pr)
    target_pr = np.array(target_pr)
    max_target_val = target_pr.max()
    predict_sort_ind = (-predict_pr).argsort()
    select_failure_case_num = 0
    select_failure_set = []
    critical_failure_set = []
    
    worst_case_num = 0
    worst_case_num_predicted = 0.
    
    ind = 0
    for i in target_pr:
        if i > max_target_val * 0.8:
            worst_case_num += 1
            critical_failure_set.append(ind)
        ind += 1

    for i in range(val_num):
        if i / val_num > ratio_lowerbound:
            if (predict_classify[predict_sort_ind[i]] < 1e-3 and (predict_pr[predict_sort_ind[0]] - predict_pr[predict_sort_ind[i]]) / predict_pr[predict_sort_ind[0]] > 0.05) or (predict_classify[predict_sort_ind[i]] < threshold_classify and (predict_pr[predict_sort_ind[0]] - predict_pr[predict_sort_ind[i]]) / predict_pr[predict_sort_ind[0]] > 0.3):
                continue
        
        max_predict_val = max(max_predict_val, target_pr[predict_sort_ind[i]])
        if target_pr[predict_sort_ind[i]] > max_target_val - 1e-3:
            flag = True

        if target_pr[predict_sort_ind[i]] > 0.8 * max_target_val:
            worst_case_num_predicted += 1

        select_failure_case_num += 1
        select_failure_set.append(predict_sort_ind[i])
        if select_failure_case_num / val_num > ratio_upperbound and ((predict_pr[predict_sort_ind[0]] - predict_pr[predict_sort_ind[i]]) / predict_pr[predict_sort_ind[0]] > gap):
            break
    ae = max_target_val - max_predict_val
    re = ae / max_target_val
    return flag, worst_case_num_predicted/worst_case_num, ae, re, select_failure_case_num, val_num, select_failure_set, critical_failure_set




if __name__ == "__main__":
    # arguments
    args = get_arg()
    device = 'cuda' if args.use_cuda and th.cuda.is_available() else 'cpu'
    print("device:", device)
    
    classify_dir = args.detect_classify_model_dir
    pr_dir = args.detect_normal_model_dir
    failure_type = args.failure_type

    # model parameter
    node_features_dim = args.node_features_dim
    hidden_units = args.hidden_units 
    local_heads = args.global_heads
    global_heads = args.local_heads
    output_units = 1

    classify_model = ResGAT(node_features_dim, output_units, hidden_units, local_heads, global_heads)
    classify_model.load_state_dict(th.load("./models/%s/model.th" %(classify_dir))) # load pretrain model parameters
    classify_model.to(device)
    classify_model.eval()

    pr_model = ResGAT(node_features_dim, output_units, hidden_units, local_heads, global_heads)
    pr_model.load_state_dict(th.load("./models/%s/model.th" %(pr_dir))) # load pretrain model parameters
    pr_model.to(device)
    pr_model.eval()

    
    print("loading dataset...")
    eval_data_dir = args.eval_data_dir
    eval_data_dict = load_datas("./data/dataset/"+eval_data_dir)
    
    activator = nn.Sigmoid() # for classification
    
    
    # Inference
    pr_model.eval()
    classify_model.eval()
    threshold_classify = 0.1

    res = []
    case_num = 0
    # evaluation results for robust validation
    topk_accnum = 0 
    topk_ae = []
    topk_re = []
    topk_mre_topo = []
    topk_select_ratio = []
    speedup_ratios = [] 
    topk_accuracy = []
    topk_worst_accuracy = [] # accuracy to include failure cases which pr / max_pr > 0.95
    # num_k = 0
    selected_failure_num_total = 0.
    failure_num_total = 0.
    
    cleanup_dir("./failure_detect/")
    inf_log_file = open("./failure_detect/%s.log" % (eval_data_dir), "w")



    for topo_name in eval_data_dict:
        failure_case_file = open("./failure_set/%s.json" % (topo_name), "w")
        case_num_topo = 0
        topk_accnum_topo = 0
        selected_failure_num_topo = 0.
        failure_num_topo = 0.
        topk_re_topo = 0.
        print("Evaluate Topo Name:", topo_name)
        print("Evaluate Topo Name:", topo_name, file=inf_log_file)
        data_list = []
        eval_data = eval_data_dict[topo_name]
        eval_data_generator = gnn_data_generator(eval_data['node_num'], eval_data['link_num'], eval_data['linkSet'], node_features_dim, eval_data['edge_index'], eval_data['datas'], 'classify', failure_type=failure_type, shuffle=False, sample=False) 
        for data in eval_data_generator:
            data_list.append(data)
        eval_loader = DataLoader(data_list, batch_size=1)
        
        for batch in eval_loader:
            batch.to(device)
            case_num += 1
            case_num_topo += 1
            with th.no_grad():
                start = time.time()
                y_classify = activator(classify_model(batch))
                y_pr = pr_model(batch)
                end = time.time()
                print("time overhead:", end - start)
                
                ret_classify = y_classify.cpu().numpy().flatten().tolist()
                ret_pr = y_pr.cpu().numpy().flatten().tolist()
                re_pr = ((y_pr.detach().squeeze(-1) - batch.prs.detach()) / (1e-3 + batch.prs.detach())).cpu().numpy().flatten().tolist()
                label_classify = batch.y.cpu().numpy().flatten().tolist()
                label_pr = batch.prs.cpu().numpy().flatten().tolist()
                mask =  (batch.mask / (batch.mask + 1e-9)).cpu().numpy().flatten().tolist()

                predict_classify = []
                predict_pr = []
                target_classify = []
                target_pr = []
                failure_set = batch.failure_set[0]
                for i in range(len(ret_pr)):
                    if mask[i] != 0:
                        predict_classify.append(ret_classify[i])
                        predict_pr.append(ret_pr[i])
                        target_classify.append(label_classify[i])
                        target_pr.append(label_pr[i])
                        res.append(re_pr[i])
                

                compare_list = []
                sorted_index = (-np.array(target_pr)).argsort()
                for i in sorted_index:
                    compare_list.append((predict_pr[i], target_pr[i], predict_classify[i], target_classify[i]))
                print("compare_list:", compare_list, file=inf_log_file)
                
                flag, accuracy, ae, re, selected_failure_num, failure_num, select_failure_case, critical_failure_case = calc_gap_accuracy(predict_pr, target_pr, predict_classify, threshold_classify, 0.2, 0.01, 0.15)
                if re > 0.:
                    print("ind:", case_num, file=inf_log_file)
                
                select_failure_set = [failure_set[i] for i in select_failure_case]
                critical_failure_set = [failure_set[i] for i in critical_failure_case]
                target_sort_ind = (-np.array(target_pr)).argsort()
                sorted_failure_set = [failure_set[i] for i in target_sort_ind]
                predict_sort_ind = (-np.array(predict_pr)).argsort()
                sorted_failure_set_predict = [failure_set[i] for i in predict_sort_ind]
                print(max(target_pr), file=failure_case_file)
                print(json.dumps(batch.link_capas[0]), file=failure_case_file)
                print(json.dumps(batch.TM[0]), file=failure_case_file)
                print(json.dumps(sorted_failure_set), file=failure_case_file)
                print(json.dumps(select_failure_set), file=failure_case_file)
                print(json.dumps(sorted_failure_set_predict), file=failure_case_file)
                print(json.dumps(critical_failure_set), file=failure_case_file)
                
                selected_failure_num_total += selected_failure_num
                selected_failure_num_topo += selected_failure_num
                failure_num_total += failure_num
                failure_num_topo += failure_num
                print("accurate flag:", flag, "topk ae:", ae, "topk re:", re, "accuracy:", accuracy, "select_failure_case_ratio:", selected_failure_num/failure_num, "critical failure ratio:", sum(target_classify)/failure_num, file=inf_log_file)
                topk_re.append(re)
                topk_ae.append(ae)
                topk_re_topo += re
                topk_worst_accuracy.append(accuracy)
                
                if flag:
                    topk_accnum += 1
                    topk_accnum_topo += 1


                
        print("Topo:", topo_name, "top k accuracy:", topk_accnum_topo / case_num_topo, "selected failure ratio:", selected_failure_num_topo / failure_num_topo, file=inf_log_file)  
        speedup_ratios.append(failure_num_topo/selected_failure_num_topo)
        topk_select_ratio.append(selected_failure_num_topo / failure_num_topo)     
        topk_accuracy.append(topk_accnum_topo / case_num_topo)
        topk_mre_topo.append(topk_re_topo / case_num_topo)
    print("topk accuracy:", topk_accnum / case_num, file=inf_log_file)
    print("Select failure ratio:", selected_failure_num_total / failure_num_total, file=inf_log_file)
    print("Speed up ratio:", max(speedup_ratios), min(speedup_ratios), sum(speedup_ratios)/len(speedup_ratios), file=inf_log_file)
    


    # evaluation results for robust validation
    sns.kdeplot(topk_ae, cumulative=True, cut=0)
    plt.plot()
    plt.savefig('./failure_detect/topk-ae-kde-%s.png' % (eval_data_dir))
    plt.clf()
    sns.kdeplot(topk_re, cumulative=True)
    plt.plot()
    plt.savefig('./failure_detect/topk-re-kde-%s.png' % (eval_data_dir))
    plt.clf()
    topk_re_str = json.dumps(topk_re)
    log_file = open('./failure_detect/%s-topk-re.json' % (eval_data_dir), "w")
    print(topk_re_str, file=log_file)

    sns.kdeplot(topk_worst_accuracy, cumulative=True)
    plt.plot()
    plt.savefig('./failure_detect/topk-worst_accuracy-kde-%s.png' % (eval_data_dir))
    topk_worst_accuracy_str = json.dumps(topk_worst_accuracy)
    log_file = open('./failure_detect/%s-topk-worst_accuracy.json' % (eval_data_dir), "w")
    print(topk_worst_accuracy_str, file=log_file)

    sns.kdeplot(topk_accuracy, cumulative=True)
    plt.plot()
    plt.savefig('./failure_detect/topk-accuracy-kde-%s.png' % (eval_data_dir))
    plt.clf()
    topk_mre_topo_str = json.dumps(topk_mre_topo)
    log_file = open('./failure_detect/%s-topk-mre.json' % (eval_data_dir), "w")
    print(topk_mre_topo_str, file=log_file)
    sns.kdeplot(topk_select_ratio, cumulative=True)
    plt.plot()
    plt.savefig('./failure_detect/topk-select_ratio-kde-%s.png' % (eval_data_dir))
    plt.clf()
    topk_select_ratio_str = json.dumps(topk_select_ratio)
    log_file = open('./failure_detect/%s-topk-select_ratio.json' % (eval_data_dir), "w")
    print(topk_select_ratio_str, file=log_file)

   
