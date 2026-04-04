import torch
import torch.nn as nn
from cbam import CBAM, ChannelAttention
from convnext_block import ConvNeXtBlock


class Model(nn.Module):
    def __init__(self, out_dim: int = 5, drop_path_rate: float = 0.065):
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

        # Stage 4: 64x4x4
        self.stage4 = nn.Sequential(                                    
            ConvNeXtBlock(32, 64, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(64, 64, drop_path_rate=drop_path_rate),
        )

        # Attention + Pooling
        self.ca      = ChannelAttention(64)
        self.gap     = nn.AdaptiveAvgPool2d(1)                          
        self.flatten = nn.Flatten(1)

        # MLP 
        self.shared = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
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

        x = self.ca(x)              
        x = self.gap(x)              
        x = self.flatten(x)    

        feat = self.shared(x)        

        xyz = self.head_xyz(feat)    
        ang = self.head_ang(feat)     

        return torch.cat([xyz, ang], dim=1)  