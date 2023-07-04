import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
from wl import WL


class MLP(nn.Module):
    def __init__(self, input_dim, nhid):
        super(MLP,self).__init__()
        self.cls = nn.Sequential(
            torch.nn.Linear(input_dim,nhid))
        
    def forward(self, features):
        output = self.cls(features)
        return output
            
class GCN(nn.Module):
    def __init__(self, input_dim, nhid, num_classes, ngl, dropout, edge_dropout, edgenet_input_dim):
        super(GCN, self).__init__()
        K=3   
        hidden = [nhid for i in range(ngl)] 
        self.dropout = dropout
        self.edge_dropout = edge_dropout 
        bias = False 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.ngl = ngl 
        self.gconv = nn.ModuleList()
        for i in range(ngl):
            in_channels = input_dim if i==0  else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias)) 
          
        self.cls = nn.Sequential(
                torch.nn.Linear(16, 128),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(128), 
                torch.nn.Linear(128, num_classes))

        self.edge_net = WL(input_dim=edgenet_input_dim//2, dropout=dropout)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight) # He init
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False): 
        if self.edge_dropout>0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0],1]).cuda()
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask] 
                edgenet_input = edgenet_input[self.bool_mask] # Weights
            
        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        

        # GCN residual connection
        # input layer
        features = F.dropout(features, self.dropout, self.training)
        x = self.relu(self.gconv[0](features, edge_index, edge_weight)) 
        x_temp = x
        
        # hidden layers
        for i in range(1, self.ngl - 1): # self.ngl→7
            x = F.dropout(x_temp, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index, edge_weight)) 
            x_temp = x_temp + x # ([871,64])

        # output layer
        x = F.dropout(x_temp, self.dropout, self.training)
        x = self.relu(self.gconv[self.ngl - 1](x, edge_index, edge_weight))
        x_temp = x_temp + x

        output = x # Final output is not cumulative
        output = self.cls(output) 
        
        return output, edge_weight
    

    

