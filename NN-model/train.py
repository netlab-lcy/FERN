from lib.dataloader import load_datas
from config.arguments import get_arg
from lib.nnmodel import ResGAT
from lib.gnn_utils import gnn_data_generator
from lib.utils import cleanup_dir, smooth
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch, Data
from torch.nn import functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time




if __name__ == "__main__":
    # arguments
    args = get_arg()
    device = 'cuda' if args.use_cuda and th.cuda.is_available() else 'cpu'
    print("device:", device)
    mode = args.mode 
    log_dir = args.log_dir
    model_load_dir = args.model_load_dir
    batch_size = args.batch_size 
    epochs = args.training_epochs
    failure_type = args.failure_type
    part_failure = args.part_failure

    # model parameter
    node_features_dim = args.node_features_dim
    hidden_units = args.hidden_units 
    local_heads = args.global_heads
    global_heads = args.local_heads
    output_units = 1

    model = ResGAT(node_features_dim, output_units, hidden_units, local_heads, global_heads)
    # load pretrain model parameters
    if model_load_dir != None:
        model.load_state_dict(th.load("./models/%s/model.th" % (model_load_dir))) 
    model.to(device)
    
    print("loading dataset...")
    train_data_dir = args.train_data_dir
    valid_data_dir = args.valid_data_dir
    
    if epochs > 0:
        train_data_dict = load_datas("./data/dataset/"+train_data_dir)
        valid_data_dict = load_datas("./data/dataset/"+valid_data_dir)
    
    
    optimizer = optim.Adam(model.parameters())
    if mode == "classify":
        activator = nn.Sigmoid()
        loss_fn = nn.BCELoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")
    
    print("start training...")
    start_time = time.time()
    loss_train = []
    loss_valid = []
    for epoch in range(epochs):
        print("epoch:", epoch)
        model.train()
        for topo_name in train_data_dict:
            data_list = []
            train_data = train_data_dict[topo_name]
            train_data_generator = gnn_data_generator(train_data['node_num'], train_data['link_num'], train_data['linkSet'], node_features_dim, train_data['edge_index'], train_data['datas'], mode, failure_type=failure_type, shuffle=False, sample=True, part_failure=part_failure)
            for data in train_data_generator:
                data_list.append(data)
            
            batch_global_edge_index = []
            for i in range(0, len(data_list), batch_size):
                batch = Batch.from_data_list([Data(x=data.x, edge_index=data.global_edge_index) for data in data_list[i : min(len(data_list), i+batch_size)]])
                batch_global_edge_index.append(batch.edge_index)
                
            train_loader = DataLoader(data_list, batch_size=batch_size)
            
            batch_ind = 0
            for batch in train_loader:
                batch.global_edge_index = batch_global_edge_index[batch_ind] # without this code the global_edge_index is also correct
                batch.to(device)
                y = model(batch)
                
                if mode == "classify":
                    y = activator(y)
                    loss = loss_fn(y, batch.y.unsqueeze(-1))
                    loss = loss * batch.mask.unsqueeze(-1)
                    loss = loss.sum() / batch.num_graphs
                else:
                    loss = loss_fn(y, batch.y.unsqueeze(-1))
                    loss = loss * batch.mask.unsqueeze(-1)
                    loss = loss.sum() / batch.num_graphs
                    # for extra loss
                    extra_loss = F.relu((y.squeeze(-1) * batch.mask2).sum()) / batch.num_graphs
                    loss = loss + extra_loss
                    
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("training loss:", loss.item())
                loss_train.append(loss.item())

                
                # adjust the weight of training data
                if mode == "normal":
                    # performance ratio of each case normalized with worstr failure case
                    ret = ((y.detach() - batch.y.unsqueeze(-1).detach()) / (1e-3 + batch.y.unsqueeze(-1).detach()) * (batch.mask / (batch.mask + 1e-9)).unsqueeze(-1)).cpu().numpy().flatten().tolist()
                    mask = batch.mask.cpu().detach().numpy().flatten().tolist()
                    ind = 0

                    reverse_ind = []
                    for it in batch.reverse_ind:
                        reverse_ind += it
                    for i in range(len(ret)):
                        if mask[i] != 0:
                            data_ind = reverse_ind[ind][1]
                            case_ind = reverse_ind[ind][0]
                            data = train_data['datas'][data_ind]
                            data['weights'][case_ind] = ret[i]
                            ind += 1
                batch_ind += 1        
        
        # model validation        
        model.eval()
        total_loss = 0.
        for topo_name in valid_data_dict:
            print("Evaluate Topo Name:", topo_name)
            data_list = []
            eval_data = valid_data_dict[topo_name]
            eval_data_generator = gnn_data_generator(eval_data['node_num'], eval_data['link_num'], eval_data['linkSet'], node_features_dim, eval_data['edge_index'], eval_data['datas'], mode, failure_type=failure_type, shuffle=False, sample=False, part_failure=part_failure)
            for data in eval_data_generator:
                data_list.append(data)

            batch_global_edge_index = []
            for i in range(0, len(data_list), batch_size):
                batch = Batch.from_data_list([Data(x=data.x, edge_index=data.global_edge_index) for data in data_list[i : min(len(data_list), i+batch_size)]])
                batch_global_edge_index.append(batch.edge_index)
            eval_loader = DataLoader(data_list, batch_size=batch_size)

            batch_ind = 0
            for batch in eval_loader:
                batch.global_edge_index = batch_global_edge_index[batch_ind]
                batch.to(device)
                with th.no_grad():
                    y = model(batch)
                    if mode == "classify":
                        y = activator(y)
                        loss = loss_fn(y, batch.y.unsqueeze(-1))
                        loss = loss * batch.mask.unsqueeze(-1)
                        loss = loss.sum() / batch.num_graphs
                    else:
                        loss = loss_fn(y, batch.y.unsqueeze(-1))
                        loss = loss * batch.mask.unsqueeze(-1)
                        loss = loss.sum() / batch.num_graphs
                        # for extra loss
                        extra_loss = F.relu((y.squeeze(-1) * batch.mask2).sum()) / batch.num_graphs
                        loss = loss + extra_loss
                    total_loss += loss

                batch_ind += 1

        loss_valid.append(total_loss)
        print('Validation loss:', total_loss)

    end_time = time.time()    
    
    # save model and log
    cleanup_dir("./models/%s" % (log_dir))
    cleanup_dir("./log/%s" % (log_dir))
    th.save(model.state_dict(), "./models/%s/model.th" % (log_dir))    
    
    loss_train_file = open("./log/%s/loss-train.log"  % (log_dir), "w")
    json.dump(loss_train, loss_train_file)
    plt.plot(smooth(loss_train, 100))
    if mode == "normal":
        plt.yscale('log')
    plt.savefig("./log/%s/train.png" % (log_dir))
    plt.clf()


    loss_test_file = open("./log/%s/loss-eval.log"  % (log_dir), "w")
    json.dump(loss_valid, loss_test_file)
    plt.plot(loss_valid)
    if mode == "normal":
        plt.yscale('log')
    plt.savefig("./log/%s/test.png" % (log_dir))
    plt.clf()

    time_log_file = open("./log/%s/time.log" % (log_dir), "w")
    print("training time cost: %f(s)" % (end_time - start_time), file=time_log_file)
    
    