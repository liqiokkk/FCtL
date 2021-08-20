from .base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .helpers import get_upsampling_weight
import torch
from itertools import chain
from .FCtL import FCtL
import numpy as np
from einops import rearrange
        
class MiniFCN8(BaseModel):
    def __init__(self, num_classes, pretrained=True):
        super(MiniFCN8, self).__init__()
        
        self.vgg = models.vgg16(pretrained)
        self.features = list(self.vgg.features.children())
        self.classifier = list(self.vgg.classifier.children())

        # Pad the input to enable small inputs and allow matching feature maps
        self.features[0].padding = (100, 100)

        # Enbale ceil in max pool, to avoid different sizes when upsampling
        for layer in self.features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True

        self.big_pool3 = nn.Sequential(*self.features[:17])
        self.big_pool4 = nn.Sequential(*self.features[17:24])
        self.big_pool5 = nn.Sequential(*self.features[24:])

    def forward(self, x):
        pool3_2 = self.big_pool3(x)
        pool4_2 = self.big_pool4(pool3_2)
        pool5_2 = self.big_pool5(pool4_2)
        
        return pool5_2

                
class FCN8(BaseModel):
    def __init__(self, num_classes, mode=1, pretrained=True, freeze_bn=False, freeze_backbone=False):
        super(FCN8, self).__init__()
        
        self.mode = mode
        
        self.vgg = models.vgg16(pretrained)
        self.features = list(self.vgg.features.children())
        self.classifier = list(self.vgg.classifier.children())

        # Pad the input to enable small inputs and allow matching feature maps
        self.features[0].padding = (100, 100)

        # Enbale ceil in max pool, to avoid different sizes when upsampling
        for layer in self.features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True
        # Extract pool3, pool4 and pool5 from the VGG net
        self.pool3 = nn.Sequential(*self.features[:17])
        self.pool4 = nn.Sequential(*self.features[17:24])
        self.pool5 = nn.Sequential(*self.features[24:])
        if self.mode == 2:     
            self.big_pool3 = nn.Sequential(*self.features[:17])
            self.big_pool4 = nn.Sequential(*self.features[17:24])
            self.big_pool5 = nn.Sequential(*self.features[24:])
        if self.mode == 2 or self.mode == 3:
            self.big_attention = FCtL(512, 512)
            if self.mode == 3:
                self.big_attention2 = FCtL(512, 512)
                self.big_attention3 = FCtL(512, 512)
                conv_nd = nn.Conv2d
                self.in_1 = conv_nd(512, 512, kernel_size=1)
                self.in_2 = conv_nd(512, 512, kernel_size=1)
                self.in_3 = conv_nd(512, 512, kernel_size=1)
                self.trans = conv_nd(512*3, 512*3, kernel_size=1)
                self.out_1 = conv_nd(512, 512, kernel_size=1)
                self.out_2 = conv_nd(512, 512, kernel_size=1)
                self.out_3 = conv_nd(512, 512, kernel_size=1)
                self.softmax_H = nn.Softmax(dim=0)
                
        # Adjust the depth of pool3 and pool4 to num_classe
        self.adj_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.adj_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        # Replace the FC layer of VGG with conv layers
        conv6 = nn.Conv2d(512, 4096, kernel_size=7)
        conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        output = nn.Conv2d(4096, num_classes, kernel_size=1)
        # Copy the weights from VGG's FC pretrained layers
        conv6.weight.data.copy_(self.classifier[0].weight.data.view(
            conv6.weight.data.size()))
        conv6.bias.data.copy_(self.classifier[0].bias.data)

        conv7.weight.data.copy_(self.classifier[3].weight.data.view(
            conv7.weight.data.size()))
        conv7.bias.data.copy_(self.classifier[3].bias.data)

        # Get the outputs
        self.output = nn.Sequential(conv6, nn.ReLU(inplace=True), nn.Dropout(),
                                    conv7, nn.ReLU(inplace=True), nn.Dropout(), 
                                    output)

        # We'll need three upsampling layers, upsampling (x2 +2) the ouputs
        # upsampling (x2 +2) addition of pool4 and upsampled output 
        # upsampling (x8 +8) the final value (pool3 + added output and pool4)
        self.up_output = nn.ConvTranspose2d(num_classes, num_classes,
                                            kernel_size=4, stride=2, bias=False)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=4, stride=2, bias=False)
        self.up_final = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=16, stride=8, bias=False)

        # We'll use guassian kernels for the upsampling weights
        self.up_output.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 4))
        self.up_pool4_out.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 4))
        self.up_final.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 16))
        # We'll freeze the wights, this is a fixed upsampling and not deconv
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.requires_grad = False
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.pool3, self.pool4, self.pool5], False)
        
    def forward(self, x, pool5_10=None, pool5_15=None, y=None):
        imh_H, img_W = x.size()[2], x.size()[3]
        # Forward the image
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        if self.mode == 2:
            pool3_10 = self.big_pool3(y)
            pool4_10 = self.big_pool4(pool3_10)
            pool5_10 = self.big_pool5(pool4_10)
        if self.mode == 2 or self.mode == 3:
            if self.mode == 2:
                att1 = self.big_attention(pool5, pool5_10)
                pool5 = pool5 + att1
            else:
                att1 = self.big_attention(pool5, pool5_10)
                att2 = self.big_attention2(pool5, pool5_15)
                att3 = self.big_attention3(pool5, pool5)
                H_1 = self.in_1(att1)
                H_2 = self.in_2(att2)
                H_3 = self.in_3(att3)
                H_cat = torch.cat((H_1, H_2, H_3), 1)
                H_tra = self.trans(H_cat)
                H_spl = torch.split(H_tra, 512, dim=1)
                H_4 = torch.sigmoid(self.out_1(H_spl[0]))
                H_5 = torch.sigmoid(self.out_2(H_spl[1]))
                H_6 = torch.sigmoid(self.out_3(H_spl[2]))
                H_st = torch.stack((H_4, H_5, H_6), 0)
                H_all = self.softmax_H(H_st)
                out = H_all[0] * att1 + H_all[1] * att2 +  H_all[2] * att3
                pool5 = pool5 + out
        output = self.output(pool5)
        # Get the outputs and upsmaple them
        up_output = self.up_output(output) #7*36*36

        # Adjust pool4 and add the uped-outputs to pool4
        adjstd_pool4 = self.adj_pool4(0.01 * pool4)
        add_out_pool4 = self.up_pool4_out(adjstd_pool4[:, :, 5: (5 + up_output.size()[2]), 
                                            5: (5 + up_output.size()[3])]
                                           + up_output)

        # Adjust pool3 and add it to the uped last addition
        adjstd_pool3 = self.adj_pool3(0.0001 * pool3)
        final_value = self.up_final(adjstd_pool3[:, :, 9: (9 + add_out_pool4.size()[2]), 9: (9 + add_out_pool4.size()[3])]
                                 + add_out_pool4)

        # Remove the corresponding padded regions to the input img size
        final_value = final_value[:, :, 31: (31 + imh_H), 31: (31 + img_W)].contiguous()
        return final_value

    def get_backbone_params(self):
        return chain(self.pool3.parameters(), self.pool4.parameters(), self.pool5.parameters(), self.output.parameters())

    def get_decoder_params(self):
        return chain(self.up_output.parameters(), self.adj_pool4.parameters(), self.up_pool4_out.parameters(),
            self.adj_pool3.parameters(), self.up_final.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

