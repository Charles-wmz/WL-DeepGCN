" Adaptive weighte-learning network "

import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch.nn.functional as F
from torch import nn

class WL(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super(WL, self).__init__()
        h1=256
        h2=128
        self.parser =nn.Sequential(
                nn.Linear(input_dim, h1, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(h1),
                nn.Dropout(dropout),
                nn.Linear(h1, h2, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm1d(h2),
                nn.Dropout(dropout),
                nn.Linear(h2, h2, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()

    def forward(self, x):
        x1 = x[:,0:self.input_dim]
        x2 = x[:,self.input_dim:]
        h1 = self.parser(x1) 
        h2 = self.parser(x2) 
        p = (self.cos(h1,h2) + 1)*0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

