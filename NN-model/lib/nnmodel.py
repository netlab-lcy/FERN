import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATv2Conv, GATConv

from torch_geometric.nn.sequential import Sequential


class MixGAT(nn.Module):
    def __init__(self, hidden_units, local_heads, global_heads):
        super(MixGAT, self).__init__()
        self.conv1 = GATv2Conv(hidden_units, hidden_units, heads=local_heads)
        self.conv2 = GATv2Conv(hidden_units, hidden_units, heads=global_heads)
        self.fc = nn.Linear(hidden_units*(local_heads + global_heads), hidden_units)
    
    def forward(self, x, edge_index, global_edge_index):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x, global_edge_index)
        x = torch.cat([x1, x2], -1)
        x = self.fc(x)
        
        return x 

class ResGAT(nn.Module):
    def __init__(self, num_node_features, output_units, hidden_units=16, local_heads=1, global_heads=1):
        super(ResGAT, self).__init__()
        
        self.conv1 = MixGAT(hidden_units, local_heads, global_heads)
        self.conv2 = MixGAT(hidden_units, local_heads, global_heads)
        
        
        self.conv3 = MixGAT(hidden_units, local_heads, global_heads)
        self.conv4 = MixGAT(hidden_units, local_heads, global_heads)
        self.conv5 = MixGAT(hidden_units, local_heads, global_heads)
        
        
        self.embed_fc = nn.Sequential(
            nn.Linear(num_node_features, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.fc2 =  nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.fc3 =  nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.fc4 =  nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.fc5 =  nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, output_units),
        )
        self.leakyrelu = nn.LeakyReLU()

        self.gatnorm1 = nn.LayerNorm(hidden_units)
        self.fcnorm1 = nn.LayerNorm(hidden_units)
        self.gatnorm2 = nn.LayerNorm(hidden_units)
        self.fcnorm2 = nn.LayerNorm(hidden_units)
        self.gatnorm3 = nn.LayerNorm(hidden_units)
        self.fcnorm3 = nn.LayerNorm(hidden_units)
        self.gatnorm4 = nn.LayerNorm(hidden_units)
        self.fcnorm4 = nn.LayerNorm(hidden_units)
        self.gatnorm5 = nn.LayerNorm(hidden_units)
        self.fcnorm5 = nn.LayerNorm(hidden_units)


    def forward(self, data):
        x, edge_index, global_edge_index = data.x, data.edge_index, data.global_edge_index
        x = x.float()
        x = self.embed_fc(x)
        
        residual = x
        x = self.conv1(x, edge_index, global_edge_index)
        x = self.gatnorm1(x + residual)

        residual = x
        x = self.fc1(x)
        x = self.fcnorm1(x + residual)
        
        residual = x
        x = self.conv2(x, edge_index, global_edge_index)
        x = self.gatnorm2(x + residual)

        residual = x
        x = self.fc2(x)
        x = self.fcnorm2(x + residual)

        residual = x
        x = self.conv3(x, edge_index, global_edge_index)
        x = self.gatnorm3(x + residual)

        residual = x
        x = self.fc3(x)
        x = self.fcnorm3(x + residual)
        
        residual = x
        x = self.conv4(x, edge_index, global_edge_index)
        x = self.gatnorm4(x + residual)

        residual = x
        x = self.fc4(x)
        x = self.fcnorm4(x + residual)

        residual = x
        x = self.conv5(x, edge_index, global_edge_index)
        x = self.gatnorm5(x + residual)

        residual = x
        x = self.fc5(x)
        x = self.fcnorm5(x + residual)

        
        y = self.fc(x)

        return y

     

 
