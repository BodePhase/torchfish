import torch
import torch.nn as nn
import torch.nn.functional as F


class KasparovPolicyNet(nn.Module):
    def __init__(self, num_classes: int = 4544, in_channels: int = 14):
        super().__init__()
        # Stem convoluzionale (conv1 primo parametro per compatibilità con inference.py)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        # Blocco Residuale 8x8 con BatchNorm2d
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Policy Head con BatchNorm1d e Dropout
        self.fc1 = nn.Linear(128 * 8 * 8, 1024, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(p=0.15)
        self.fc_out = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))

        residual = x
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        x = F.relu(out + residual)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc_out(x)