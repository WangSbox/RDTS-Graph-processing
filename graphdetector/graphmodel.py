# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, TransformerConv, DNAConv, ResGatedGraphConv
from torch_geometric.nn import GCN, GAT, GraphSAGE, PNA, BatchNorm, GraphNorm

class GCN1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.batch = BatchNorm(64)
        self.conv2 = GCNConv(64, 64)
        self.batch1 = BatchNorm(64)
        self.conv3 = GCNConv(64, 256)
        self.batch2 = BatchNorm(256)
        self.conv4 = GCNConv(256, 256)
        self.batch3 = BatchNorm(256)
        self.conv5 = GCNConv(256, 512)
        self.batch4 = BatchNorm(512)
        self.conv6 = GCNConv(512, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.batch(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv2(x, edge_index)
        x = self.batch1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv3(x, edge_index)
        x = self.batch2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv4(x, edge_index)
        x = self.batch3(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv5(x, edge_index)
        x = self.batch4(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv6(x, edge_index)
        # x = F.relu(x)

        # x = self.conv3(x, edge_index)
        # x = F.dropout(x,0.1,training=True)
        # x = F.relu(x)

        return torch.sigmoid(x)

class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(3, 64)
        self.batch = BatchNorm(64)
        self.conv2 = GCNConv(64, 64)
        self.batch1 = BatchNorm(64)
        self.conv3 = GCNConv(64, 256)
        self.batch2 = BatchNorm(256)
        self.conv4 = GCNConv(256, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.batch(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv2(x, edge_index)
        x = self.batch1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv3(x, edge_index)
        x = self.batch2(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv4(x, edge_index)

        return torch.sigmoid(x)

class GCN3_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 128)
        self.at1 = GATConv(128, 256, 8, False)
        self.nor = GraphNorm(128)
        self.batch1 = BatchNorm(128)
        self.conv2 = GCNConv(128, 256)
        self.batch2 = BatchNorm(256)
        self.conv3 = GCNConv(256, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.batch1(self.conv1(x, edge_index)), 0.1)
        # x = F.relu(self.nor(self.at1(x,edge_index)))
        x = F.leaky_relu(self.batch2(self.conv2(x, edge_index)), 0.3)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

class GCN3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gcn = GraphSAGE(3, 64, 2)
        self.gcn1 = GraphSAGE(64, 256, 2)
        self.gcn2 = GraphSAGE(256, 512, 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.batch1(self.gcn(x, edge_index)), 0.1)
        x = F.leaky_relu(self.batch2(self.gcn1(x, edge_index)), 0.1)
        x = self.gcn2(x, edge_index)
        return torch.sigmoid(x)

class GCN4(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gcn = GCN(3, 64, 2)
        self.gcn1 = GCN(64, 256, 2)
        self.gcn2 = GCN(256, 512, 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.batch1(self.gcn(x, edge_index)), 0.1)
        x = F.leaky_relu(self.batch2(self.gcn1(x, edge_index)), 0.1)
        x = self.gcn2(x, edge_index)
        return torch.sigmoid(x)

class GCN5(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gcn = GAT(3, 64, 2, heads=3)
        self.gcn1 = GAT(64, 256, 2, heads=3)
        self.gcn2 = GAT(256, 512, 2, 1, heads=3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.batch1(self.gcn(x, edge_index)), 0.1)
        x = F.leaky_relu(self.batch2(self.gcn1(x, edge_index)), 0.1)
        x = self.gcn2(x, edge_index)
        return torch.sigmoid(x)

class GCN5_1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gcn = GAT(3, 64, 2, heads=8)
        self.gcn1 = GAT(64, 256, 2, heads=8)
        self.gcn2 = GAT(256, 512, 2, 1, heads=8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.batch1(self.gcn(x, edge_index)), 0.1)
        x = F.leaky_relu(self.batch2(self.gcn1(x, edge_index)), 0.1)
        x = self.gcn2(x, edge_index)
        return torch.sigmoid(x)

class GCN6(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gcn = PNA(3, 64, 2)
        self.gcn1 = PNA(64, 256, 2)
        self.gcn2 = PNA(256, 512, 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.batch1(self.gcn(x, edge_index)), 0.1)
        x = F.leaky_relu(self.batch2(self.gcn1(x, edge_index)), 0.1)
        x = self.gcn2(x, edge_index)
        return torch.sigmoid(x)
        