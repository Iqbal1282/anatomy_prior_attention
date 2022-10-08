'''
Dynamically Weighted Anatomy Mask Attention Module
'''

import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    def __init__(self, in_channels, r=1):
        super().__init__()
        self.in_channels = in_channels
        self.r = r
        self.conv_list = nn.ModuleList()
        for i in range(4):
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(self.in_channels, int(self.in_channels * r), kernel_size=1),
                nn.BatchNorm2d(int(r * self.in_channels)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
        self.final_block = nn.Sequential(
            nn.Conv2d(int(r * self.in_channels), self.in_channels, kernel_size=1),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid(),
            nn.Dropout(p=0.1)
        )

    def forward(self, feature_map, mask):
        N, C, H, W = feature_map.shape
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False) # (N, 1, H, W)
        masked_feature_map = feature_map * mask # Spatial Attention (N, C, H, W)
        avg_pooled_fm = F.adaptive_avg_pool2d(feature_map, output_size=1) # (N, C, 1, 1)
        max_pooled_fm = F.adaptive_max_pool2d(feature_map, output_size=1) # (N, C)
        avg_pooled_mfm = F.adaptive_avg_pool2d(masked_feature_map, output_size=1) # (N, C, 1, 1)
        max_pooled_mfm = F.adaptive_max_pool2d(masked_feature_map, output_size=1) # (N, C)
        channel_weight = self.final_block(self.conv_list[0](avg_pooled_fm) + self.conv_list[1](max_pooled_fm) +\
                         self.conv_list[2](max_pooled_mfm) + self.conv_list[3](avg_pooled_mfm))
        channel_weight =  channel_weight.view(N, C, 1, 1)
        feature_map = channel_weight * masked_feature_map+ (1 - channel_weight) * feature_map
        #feature_map = channel_weight * feature_map
        return feature_map
