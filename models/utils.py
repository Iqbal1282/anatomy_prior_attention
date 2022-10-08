'''
init_weight_leaky -> For initializing modules with leaky-relu activation
For sigmoid -> xavier
For relu -> kaiming (maybe default)
'''

import torch.nn as nn


def init_weight_leaky(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

        if m.bias is not None: 
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)