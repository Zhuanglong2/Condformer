
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..builder import HEADS
from ...ops.deform_conv_v2 import DeformConv2d


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

@HEADS.register_module
class CtnetHead(nn.Module):
    def __init__(self, heads, channels_in, train_cfg=None, test_cfg=None, down_ratio=4, final_kernel=1, head_conv=256, branch_layers=0):
        super(CtnetHead, self).__init__()
        # self.heads = ['params']
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            # classes = 134
            if head_conv > 0:
                if 'hm' in head:
                    fc = nn.Sequential(
                      nn.Conv2d(channels_in, head_conv,kernel_size=3, padding=1, bias=True),
                      # DeformConv2d(inc=head_conv, outc=head_conv, kernel_size=3, padding=1, bias=True),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(head_conv, classes,kernel_size=final_kernel, stride=1,padding=final_kernel // 2, bias=True))
                else:
                    fc = nn.Sequential(
                        nn.Conv2d(channels_in, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2,bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels_in, classes,
                  kernel_size=final_kernel, stride=1,
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]#4 64 20 50
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z
    
    def init_weights(self):
        # ctnet_head will init weights during building
        pass


class CtnetHead_mid(nn.Module):
    def __init__(self, heads, channels_in, train_cfg=None, test_cfg=None, down_ratio=4, final_kernel=1, head_conv=256,
                 branch_layers=0):
        super(CtnetHead_mid, self).__init__()

        # self.heads = ['params']
        self.heads = ['hm']
        for head in self.heads:
            # classes = self.heads[head]
            classes = 1
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels_in, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2,
                              bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
                # fc = nn.Sequential(
                #     nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
                #     nn.Conv2d(channels_in, head_conv, kernel_size=3, padding=1, bias=True),
                #     nn.ReLU(inplace=True),
                #     nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2,
                #               bias=True))
                # if 'hm' in head:
                #     fc[-1].bias.data.fill_(-2.19)
                # else:
                #     fill_fc_weights(fc)
            else:
                fc =  nn.Sequential(
                    nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
                    nn.Conv2d(channels_in, classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]  # 4 64 20 50
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z

    def init_weights(self):
        # ctnet_head will init weights during building
        pass
