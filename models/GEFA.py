import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv
import time

class ResidualBlock(torch.nn.Module):
    def __init__(self, outfeature):
        super(ResidualBlock, self).__init__()
        self.outfeature = outfeature
        self.gcn = GCNConv(outfeature,outfeature)
        self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.gcn.weight)

    def forward(self, x, edge_index):
        identity = x
        out = self.gcn(x, edge_index)
        out = self.relu(out)
        out = self.ln(out)
        out += identity
        out = self.relu(out)
        return out

# GCN model
# GCN based model
class GEFA(torch.nn.Module):
    def __init__(self, num_features_xd, num_features_xt,
                     latent_dim=64, dropout=0.2, n_output=1, device='cpu', **kwargs):
        super(GEFA, self).__init__()

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(0.5)
        self.device = device
        self.num_rblock = 4

        # SMILES graph branch
        self.conv1_xd = GCNConv(num_features_xd, num_features_xd)
        self.conv2_xd = GCNConv(num_features_xd, num_features_xd * 2)
        self.rblock_xd = ResidualBlock(num_features_xd*2)
        self.fc_g1_d = torch.nn.Linear(num_features_xd*2, 1024)
        self.fc_g2_d = torch.nn.Linear(1024, num_features_xt)
        self.fc_g3_d = torch.nn.Linear(num_features_xt, latent_dim * 2)

        # attention
        self.first_linear = torch.nn.Linear(num_features_xt, num_features_xt)
        self.second_linear = torch.nn.Linear(num_features_xt, 1)

        # protein graph branch
        self.conv1_xt = GCNConv(num_features_xt, latent_dim)
        self.conv2_xt = GCNConv(latent_dim, latent_dim * 2)
        self.rblock_xt = ResidualBlock(latent_dim * 2)
        self.fc_g1_t = torch.nn.Linear(latent_dim * 2, 1024)
        self.fc_g2_t = torch.nn.Linear(1024, latent_dim * 2)

        self.fc1 = nn.Linear(4 * latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)


    def forward(self, drug, prot):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x2, edge_index2, batch2, prot_lens, edge_attr2 = prot.x, prot.edge_index, prot.batch, prot.prot_len, prot.edge_attr
        # drug branch
        x = self.conv1_xd(x, edge_index)
        x = self.relu(x)
        x = self.conv2_xd(x, edge_index)
        x = self.relu(x)
        for i in range(self.num_rblock):
            x = self.rblock_xd(x, edge_index)
        x = gmp(x, batch)       # global max pooling
        # flatten
        x = self.relu(self.fc_g1_d(x))
        x = self.dropout(x)
        x = self.fc_g2_d(x)
        x = self.dropout(x)
        x_changedim = self.relu(self.fc_g3_d(x))

        # protein branch
        dense_node, bool_node = to_dense_batch(x2, batch2) # (batch, num_node, num_feat) the num_node is the max node, and is padded
        cur_idx = -1
        cur_batch = 0
        # mask to remove drug node out of protein graph later
        mask = torch.ones(batch2.size(0), dtype=torch.bool)
        for size in prot_lens:
            batch_dense_node = dense_node[cur_batch]
            masked_batch_dense_node = batch_dense_node[bool_node[cur_batch]][:-1]
            node_att = F.tanh(self.first_linear(masked_batch_dense_node))
            node_att = self.dropout1(node_att)
            node_att = self.second_linear(node_att)
            node_att = self.dropout1(node_att)
            node_att = node_att.squeeze()
            node_att = F.softmax(node_att, 0)
            cur_idx += size+1
            idx_target = (edge_index2[0] == cur_idx).nonzero()
            edge_attr2[idx_target.squeeze()] = node_att.squeeze()
            idx_target = (edge_index2[1] == cur_idx).nonzero()
            edge_attr2[idx_target.squeeze()] = node_att.squeeze()
            x2[cur_idx] = x[cur_batch]
            mask[cur_idx] = False
            cur_batch += 1
        # mask to get back drug node from protein graph later
        mask_drug = ~mask
        # protein feed forward
        x2 = self.conv1_xt(x2, edge_index2, edge_attr2)
        x2 = self.relu(x2)
        x2 = self.conv2_xt(x2, edge_index2, edge_attr2)
        x2 = self.relu(x2)
        for i in range(self.num_rblock):
            x2 = self.rblock_xt(x2, edge_index2)
        x2_nodrug = x2[mask]
        batch2_nodrug = batch2[mask]
        drug_after = x2[mask_drug]
        # global max pooling
        x2 = gmp(x2_nodrug, batch2_nodrug)
        # flatten
        x2 = self.relu(self.fc_g1_t(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2_t(x2)
        x2 = self.dropout(x2)

        x = x_changedim.unsqueeze(2)
        drug_after = drug_after.unsqueeze(2)
        x = torch.cat((drug_after, x), 2)
        x = torch.max_pool1d(x, 2, 1)
        x = x.squeeze(2)

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

