import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gep


class ResidualBlock(torch.nn.Module):
    def __init__(self, outfeature):
        super(ResidualBlock, self).__init__()
        self.outfeature = outfeature
        self.gcn = GCNConv(outfeature,outfeature)
        self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
        self.elu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.gcn.weight)

    def forward(self, x, edge_index):
        identity = x
        out = self.gcn(x, edge_index)
        out = self.elu(out)
        out = self.ln(out)
        out += identity
        out = self.elu(out)
        return out



# GCN model
# GCN based model
class GLFA(torch.nn.Module):
    def __init__(self, num_features_xd, num_features_xt,
                     latent_dim=256, dropout=0.2, n_output=1, device='cpu', **kwargs):
        super(GLFA, self).__init__()

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.max_gcn_block = 4

        # drug layers
        self.conv1_1 = GCNConv(num_features_xd,num_features_xd)
        self.conv2_1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.rblock_1 = ResidualBlock(num_features_xd * 2)
        self.fc_g1_1 = torch.nn.Linear(num_features_xd * 2, 1024)
        self.fc_g2_1 = torch.nn.Linear(1024, latent_dim)

        # protein layers
        self.conv1_2 = GCNConv(num_features_xt,num_features_xd)
        self.conv2_2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.rblock_2 = ResidualBlock(num_features_xd * 2)
        self.fc_g1_2 = torch.nn.Linear(num_features_xd * 2, 1024)
        self.fc_g2_2 = torch.nn.Linear(1024, latent_dim)

        # combined layers
        self.fc1 = nn.Linear(2*latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, drug, prot):
        # drug = data[0]
        # prot = data[1]
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x2, edge_index2, batch2 = prot.x, prot.edge_index, prot.batch
        # drug graph
        x = self.conv1_1(x, edge_index)
        x = self.conv2_1(x, edge_index)
        for _ in range(self.max_gcn_block):
            x = self.rblock_1(x, edge_index)
        x = gmp(x, batch)       # global max pooling
        # flatten
        x = self.relu(self.fc_g1_1(x))
        x = self.dropout(x)
        x = self.fc_g2_1(x)
        x = self.dropout(x)

        # protein graph
        x2 = self.conv1_2(x2, edge_index2)
        x2 = self.conv2_2(x2, edge_index2)
        for _ in range(self.max_gcn_block):
            x2 = self.rblock_2(x2, edge_index2)
        x2 = gmp(x2, batch2)       # global max pooling
        # flatten
        x2 = self.relu(self.fc_g1_2(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2_2(x2)
        x2 = self.dropout(x2)

        # concat
        xc = torch.cat((x, x2), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

