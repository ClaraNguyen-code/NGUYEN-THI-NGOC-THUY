# === File: model_stgcn_origin.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A_size):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels * A_size)

    def forward(self, x, edge_index):
        # x: [B, T, V, C], edge_index: [2, E]
        B, T, V, C = x.shape
        x = x.reshape(B * T, V, C)
        x = self.gcn(x, edge_index).relu()
        x = x.reshape(B, T, -1)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x

class STGCN_Model(nn.Module):
    def __init__(self, in_channels, num_class, edge_index, num_keypoints=17):
        super().__init__()
        self.edge_index = edge_index
        self.num_keypoints = num_keypoints

        self.block1 = STGCNBlock(in_channels, 64, num_keypoints)
        self.block2 = STGCNBlock(64, 128, num_keypoints)
        self.block3 = STGCNBlock(128, 256, num_keypoints)

        self.pool = nn.AdaptiveAvgPool1d(1)  # temporal pooling
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x: [B, T, V*C] → reshape lại thành (B, T, V, C)
        B, T, VC = x.shape
        x = x.view(B, T, self.num_keypoints, -1)   # (B, T, 17, 2)

        x = self.block1(x, self.edge_index)        # (B, T, 17*64)
        x = x.view(B, T, self.num_keypoints, -1)   # (B, T, 17, 64)

        x = self.block2(x, self.edge_index)        # (B, T, 17*128)
        x = x.view(B, T, self.num_keypoints, -1)   # (B, T, 17, 128)

        x = self.block3(x, self.edge_index)        # (B, T, 17*256)
        x = x.view(B, T, self.num_keypoints, -1)   # (B, T, 17, 256)

        x = x.mean(dim=2)                          # (B, T, 256)
        x = x.transpose(1, 2)                      # (B, 256, T)
        x = self.pool(x).squeeze(-1)               # (B, 256)
        x = self.fc(x)                             # (B, num_class)
        return x
