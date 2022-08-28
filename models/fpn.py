'''
Feature Pyramid Networks

FPN -> Original Paper Implementation -> Single layer implementation
FPN_V2 -> YoloV3 FPN Implementation (concatenation instead of addition for feature merging) -> 3 layer implementation (NEED TO REFACTOR)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import init_weight_leaky


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        self.apply(init_weight_leaky)

    def forward(self, small_feat, large_feat):
        modified_small_feat = F.interpolate(small_feat, scale_factor=(2, 2), mode="bilinear", align_corners=False)
        modified_large_feat = self.weight(large_feat)

        return modified_small_feat + modified_large_feat


class FPN_V2(nn.Module):
    def __init__(self, s_channels, l_channels_in, l_channels_out):
        '''
        s, l, m with respect to dim (height or width)
        '''
        super(FPN_V2, self).__init__()
        self.s_channels = s_channels
        self.l_channels_in = l_channels_in
        self.l_channels_out = l_channels_out

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.l_channels_in, self.l_channels_out, kernel_size=1),
            nn.BatchNorm2d(self.l_channels_out),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, feat_s, feat_l):
        feat_s = F.interpolate(feat_s, scale_factor=2, mode="bilinear", align_corners=False)
        feat_l = self.conv_block(feat_l)
        return torch.cat((feat_s, feat_l), dim=1)