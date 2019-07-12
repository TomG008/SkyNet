from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from region_loss import RegionLoss
from utils import *
from collections import OrderedDict


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view([B, C, H//hs, hs, W//ws, ws]).transpose(3, 4).contiguous()
        x = x.view([B, C, H//hs*W//ws, hs*ws]).transpose(2, 3).contiguous()
        x = x.view([B, C, hs*ws, H//hs, W//ws]).transpose(1, 2).contiguous()
        x = x.view([B, hs*ws*C, H//hs, W//ws])
        return x


class SkyNet(nn.Module):
    def __init__(self):
        super(SkyNet, self).__init__()
        self.width = int(320)
        self.height = int(160)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        self.reorg = ReorgLayer(stride=2)
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        self.model_p1 = nn.Sequential(
            conv_dw( 3,  48, 1),    #dw1
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 48,  96, 1),   #dw2
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96, 192, 1),   #dw3
        )    
        self.model_p2 = nn.Sequential(    
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(192, 384, 1),   #dw4
            conv_dw(384, 512, 1),   #dw5
        )
        self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
            conv_dw(1280, 96, 1),
            nn.Conv2d(96, 10, 1, 1,bias=False),
        )
        self.loss = RegionLoss([1.4940052559648322, 2.3598481287086823,4.0113013115312155,5.760873975661669],2)
        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
        self._initialize_weights()
    def forward(self, x):
        x_p1 = self.model_p1(x)
        x_p1_reorg = self.reorg(x_p1)
        x_p2 = self.model_p2(x_p1)
        x_p3_in = torch.cat([x_p1_reorg, x_p2], 1)
        x = self.model_p3(x_p3_in)
        return x   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    
