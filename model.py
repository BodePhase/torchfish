import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessValueNetwork(nn.Module):
    def __init__(self, num_blocks=5, num_channels=128):
        super(ChessValueNetwork, self).__init__()
        
        # Initial Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(14, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([ResNetBlock(num_channels) for _ in range(num_blocks)])
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1), # Compressione a 1 canale
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh() # Output confinato in [-1, 1]
        )

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.value_head(x)