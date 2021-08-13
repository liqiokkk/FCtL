import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math

class _FCtL(nn.Module):

    def __init__(self, inplanes, planes, lr_mult, weight_init_scale):

        conv_nd = nn.Conv2d
        bn_nd = nn.BatchNorm2d

        super(_FCtL, self).__init__()

        self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.weight_init_scale = weight_init_scale
        
        self.reset_parameters()
        self.reset_lr_mult(lr_mult)
        self.reset_weight_and_weight_decay()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def reset_weight_and_weight_decay(self):
        init.normal_(self.conv_query.weight, 0, 0.01*self.weight_init_scale)
        init.normal_(self.conv_key.weight, 0, 0.01*self.weight_init_scale)
        self.conv_query.weight.wd=0.0
        self.conv_query.bias.wd=0.0
        self.conv_key.weight.wd=0.0
        self.conv_key.bias.wd=0.0

    def forward(self, x, y):

        value = self.conv_value(y)
        value = value.view(value.size(0), value.size(1), -1)
        out_sim = None
            
        query = self.conv_query(x)
        key = self.conv_key(y)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)

        key_mean = key.mean(2).unsqueeze(2)
        query_mean = query.mean(2).unsqueeze(2)
        key -= key_mean
        query -= query_mean

        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = self.softmax(sim_map)
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = out_sim.transpose(1, 2)
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim
        
        return out_sim

class FCtL(_FCtL):
    def __init__(self, inplanes, planes, lr_mult=None, weight_init_scale=1.0):
        super(FCtL, self).__init__(inplanes=inplanes, planes=planes, lr_mult=lr_mult, weight_init_scale=weight_init_scale)
