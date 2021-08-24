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
        self.conv_value_1 = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_value_2 = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
        self.conv_out = None


        self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_query_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_1 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_query_2 = conv_nd(inplanes, planes, kernel_size=1)
        self.conv_key_2 = conv_nd(inplanes, planes, kernel_size=1)
        
        self.in_1 = conv_nd(512, 512, kernel_size=1)
        self.in_2 = conv_nd(512, 512, kernel_size=1)
        self.in_3 = conv_nd(512, 512, kernel_size=1)
        self.trans = conv_nd(512*3, 512*3, kernel_size=1)
        self.out_1 = conv_nd(512, 512, kernel_size=1)
        self.out_2 = conv_nd(512, 512, kernel_size=1)
        self.out_3 = conv_nd(512, 512, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=2)
        self.softmax_H = nn.Softmax(dim=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
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

    def forward(self, x, y=None, z=None):
        residual = x

        value = self.conv_value(y)
        value = value.view(value.size(0), value.size(1), -1)
        out_sim = None
        if z is not None:
            value_1 = self.conv_value_1(z)
            value_1 = value_1.view(value_1.size(0), value_1.size(1), -1)
            out_sim_1 = None
            value_2 = self.conv_value_2(x)
            value_2 = value_2.view(value_2.size(0), value_2.size(1), -1)
            out_sim_2 = None
        
        
        query = self.conv_query(x)
        key = self.conv_key(y)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        if z is not None:
            query_1 = self.conv_query_1(x)
            key_1 = self.conv_key_1(z)
            query_1 = query_1.view(query_1.size(0), query_1.size(1), -1)
            key_1 = key_1.view(key_1.size(0), key_1.size(1), -1)
            query_2 = self.conv_query_2(x)
            key_2 = self.conv_key_2(x)
            query_2 = query_2.view(query_2.size(0), query_2.size(1), -1)
            key_2 = key_2.view(key_2.size(0), key_2.size(1), -1)


        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = self.softmax(sim_map)
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = out_sim.transpose(1, 2)
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim
        if z is not None:
            sim_map_1 = torch.bmm(query_1.transpose(1, 2), key_1)
            sim_map_1 = self.softmax(sim_map_1)
            out_sim_1 = torch.bmm(sim_map_1, value_1.transpose(1, 2))
            out_sim_1 = out_sim_1.transpose(1, 2)
            out_sim_1 = out_sim_1.view(out_sim_1.size(0), out_sim_1.size(1), *x.size()[2:])
            out_sim_1 = self.gamma_1 * out_sim_1
            sim_map_2 = torch.bmm(query_2.transpose(1, 2), key_2)
            sim_map_2 = self.softmax(sim_map_2)
            out_sim_2 = torch.bmm(sim_map_2, value_2.transpose(1, 2))
            out_sim_2 = out_sim_2.transpose(1, 2)
            out_sim_2 = out_sim_2.view(out_sim_2.size(0), out_sim_2.size(1), *x.size()[2:])
            out_sim_2 = self.gamma_2 * out_sim_2


        if z is not None:
            H_1 = self.in_1(out_sim)
            H_2 = self.in_2(out_sim_1)
            H_3 = self.in_3(out_sim_2)
            H_cat = torch.cat((H_1, H_2, H_3), 1)
            H_tra = self.trans(H_cat)
            H_spl = torch.split(H_tra, 512, dim=1)
            H_4 = torch.sigmoid(self.out_1(H_spl[0]))
            H_5 = torch.sigmoid(self.out_2(H_spl[1]))
            H_6 = torch.sigmoid(self.out_3(H_spl[2]))
            H_st = torch.stack((H_4, H_5, H_6), 0)
            H_all = self.softmax_H(H_st)
        if z is not None:
            out = residual + H_all[0] * out_sim + H_all[1] * out_sim_1 +  H_all[2] * out_sim_2
        else:
            out = residual + out_sim
        return out


class FCtL(_FCtL):
    def __init__(self, inplanes, planes, lr_mult=None, weight_init_scale=1.0):
        super(FCtL, self).__init__(inplanes=inplanes, planes=planes, lr_mult=lr_mult, weight_init_scale=weight_init_scale)
