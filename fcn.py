import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM
from cbam import ChannelAttention
from convnext_block import ConvNeXtBlock


class FCN(nn.Module):
    def __init__(self, out_dim: int = 5, drop_path_rate: float = 0.14):
        super().__init__()

        # Stage 1: 8x8x8
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Stage 2: 16x8x8
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(8,  16, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(16, 16, drop_path_rate=drop_path_rate),
        )
        self.cbam = CBAM(16)

        # Stage 3: 32x4x4
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(16, 32, stride=2, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(32, 32, drop_path_rate=drop_path_rate),
        )

        # Stage 4: 64x1x1
        self.stage4 = ConvNeXtBlock(32, 64, drop_path_rate=drop_path_rate)

        # Attention 
        self.ca   = ChannelAttention(64)
        self.gap  = nn.AdaptiveAvgPool2d(1)   

        # Shared MLP
        self.shared = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.head_xyz = nn.Linear(64, 3)

        self.head_ang = nn.Sequential(
            nn.Linear(64, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.stage1(x)

        x = self.stage2(x)
        x = self.cbam(x)

        x = self.stage3(x)
        x = self.stage4(x)

        # Attention + GAP + Norm
        x = self.ca(x)
        x = self.gap(x).flatten(1)   # (B, 64)

        feat = self.shared(x)

        xyz = self.head_xyz(feat)
        ang = self.head_ang(feat)

        return torch.cat([xyz, ang], dim=1)  # (B, 5)
    
