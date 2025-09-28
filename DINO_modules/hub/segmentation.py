import torch
import torch.nn as nn

class SegHead(nn.Module):
    def __init__(self, in_channels=1536, num_classes=21):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes

        self.bn = nn.BatchNorm2d(self.in_channels) # The Name should match the setting of mmcv
        self.conv_seg = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1) # The Name should match the setting of mmcv

    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        x = self.bn(x)
        out = self.conv_seg(x)
        return out