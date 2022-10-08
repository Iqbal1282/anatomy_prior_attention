import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
# from backbone import Densenet as DnetBackbone


def init_weight_leaky(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

        if m.bias is not None: 
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


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
        feature_map = channel_weight * masked_feature_map + (1 - channel_weight) * feature_map
        return feature_map


class DesnseNet121(nn.Module):
    '''densenet + attention module(lung + abnorm mask'''
    def __init__(self, num_classes):
        super(DesnseNet121, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.densenet121(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifiers = nn.ModuleList()
        self.lung_attention_module = MaskedAttention(1024)
        self.attention_modules = nn.ModuleList()
        for i in range(num_classes):
            self.attention_modules.append(MaskedAttention(1024))
            self.classifiers.append(nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1))
            nn.init.xavier_normal_(self.classifiers[i].weight)
        

    def forward(self, x, abnorm_mask, lung_mask):
        logit_maps = []
        logits = []

        feature_map = self.backbone(x)
        lung_attention_map = self.lung_attention_module(feature_map, lung_mask) # (N, 1024, H, W)
        for i in range(self.num_classes):
            attention_map = self.attention_modules[i](feature_map, abnorm_mask[:, i, :, :].unsqueeze(dim=1)) # (N, 1024, H, W)
            attention_map = torch.cat((attention_map, lung_attention_map), dim=1)  # (N, 2048, H, W)
            logit_map = self.classifiers[i](attention_map) # (N, 1, H, W)
            logit_maps.append(logit_map)
            out = self.pool(attention_map)
            out = F.dropout(out, p=0.2, training=self.training)
            out = self.classifiers[i](out)
            logits.append(out.squeeze(dim=-1).squeeze(dim=-1))

        logits = torch.cat(logits, dim=-1)
        logit_maps = torch.cat(logit_maps, dim=1)

        return logits, logit_maps

