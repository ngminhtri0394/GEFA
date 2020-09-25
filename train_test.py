import datetime
import os
import pickle
import sys
from math import sqrt

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from scipy import stats
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import Batch
import copy

from models.GEFA import GEFA
from models.GLFA import GLFA
from metrics import *
from graph_conversion import *
num_feat_xp = 0
num_feat_xd = 0

is_seq_in_graph = True
is_con_in_graph = True
is_profile_in_graph = True
is_emb_in_graph = True

model_name_seq = '_seq' if is_seq_in_graph is True else ''
model_name_con = '_con' if is_con_in_graph is True else ''
model_name_profile = '_pf' if is_profile_in_graph is True else ''
model_name_emb = '_emb' if is_emb_in_graph is True else ''

print('Using features: ')
print('Sequence.\n') if is_seq_in_graph
print('Contact.\n') if is_con_in_graph
print('SS + SA.\n') if is_profile_in_graph
print('Embedding.\n') if is_emb_in_graph

dataset = ['davis', 'kiba'][int(sys.argv[1])]
print('Dataset: ', dataset)

modeling = [GEFA,
            GLFA][
    int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(sys.argv[3])
print('CUDA name:', cuda_name)

set_num = int(sys.argv[4])
settings = ['_setting_1', '_setting_2', '_setting_3', '_setting_4']
setting = settings[set_num]
print("Setting: ", setting)


TRAIN_BATCH_SIZE = 128
print('Train batch size: ', TRAIN_BATCH_SIZE)
TEST_BATCH_SIZE = 256
print('Test batch size: ', TEST_BATCH_SIZE)

LR = float(sys.argv[5])
print("Learning rate: ", LR)

from_resume = [False, True][int(sys.argv[6])]

NUM_EPOCHS = 2000
print('Number of epoch: ', NUM_EPOCHS)
LOG_INTERVAL = 20


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


class GraphPairDataset(Dataset):
    def __init__(self, smile_list, prot_list, dta_graph):
        self.smile_list = smile_list
        self.prot_list = prot_list
        self.dta_graph = dta_graph

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smile = self.smile_list[idx]
        prot = self.prot_list[idx]
        GCNData_Prot, GCNData_Smile = self.dta_graph[(prot, smile)]
        return GCNData_Smile, GCNData_Prot


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        drug = data[0].to(device)
        prot = data[1].to(device)
        optimizer.zero_grad()
        output = model(drug, prot)
        # print('ouput')
        # print(output)
        loss = loss_fn(output, drug.y.view(-1, 1).float().to(device))
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    print('Average loss: {:.4f}'.format(total_train_loss / (batch_idx + 1)))
    return total_train_loss / (batch_idx + 1)


def adjust_learning_rate(optimizer, LR, scale=0.7):
    """Sets the learning rate to the initial LR decayed by 10 every interval epochs"""
    lr = LR * scale

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, filename=''):
    torch.save(state, filename)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            drug = data[0].to(device)
            prot = data[1].to(device)
            output = model(drug, prot)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, drug.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def resume(model, optimizer, savefile):
    if os.path.isfile(savefile):
        print("Loading checkpoint '{}'..".format(savefile))
        # checkpoint = torch.load(args.resume, map_location=device)
        checkpoint = torch.load(savefile)
        epoch = checkpoint['epoch'] - 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        best_mse = checkpoint['best_mse']
        best_ci = checkpoint['best_ci']
        LR = checkpoint['LR']
        print("Checkpoint loaded . Resume training from epoch {}, LR = {}.".format(epoch, LR))
        return best_mse, best_ci, epoch, optimizer, model, LR


three_one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
             'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E',
             'TYR': 'Y', 'MET': 'M'}
one_three = {three_one[e]: e for e in three_one}
smi_voc = "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTUVWYZ[\\]abcdefghilmnorstuy"
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
smi_dict = {v: i for i, v in enumerate(smi_voc)}
seq_dict = {v: i for i, v in enumerate(seq_voc)}
smi_dict_len = len(smi_dict)
seq_dict_len = len(seq_dict)
max_smi_len = 100
max_seq_len = 1000

compound_iso_smiles = []
pdbs = []
pdbs_seqs = []
all_labels = []
opts = ['train', 'test']
for opt in opts:
    df = pd.read_csv('data/'+dataset+'/split/' + dataset + '_' + opt + setting + '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
    pdbs += list(df['target_name'])
    pdbs_seqs += list(df['target_sequence'])
    all_labels += list(df['affinity'])
pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

dta_graph = {}
print(data_path)
print('Pre-processing protein')
print('Pre-processing...')
saved_prot_graph = {}
for target, seq in set(zip(pdbs, pdbs_seqs)):
    if os.path.isfile('data/'+dataset+'/map/'+ target + '.npy'):
        contactmap = np.load('data/'+dataset+'/map/' + target + '.npy')
    else:
        raise FileNotFoundError
    c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target, dataset)
    g = DATA.Data(
        x=torch.Tensor(features),
        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
        edge_attr=torch.FloatTensor(edge_weight),
        prot_len=c_size
    )
    saved_prot_graph[target] = g
saved_drug_graph = {}
for smiles in compound_iso_smiles:
    c_size2, features2, edge_index2 = smile_to_graph(smiles)
    g2 = DATA.Data(
        x=torch.Tensor(features2),
        edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
    )
    saved_drug_graph[smiles] = g2
for target, seq, smile, label in pdbs_tseqs:
    g = copy.deepcopy(saved_prot_graph[target])
    g2 = copy.deepcopy(saved_drug_graph[smile])
    g.y = torch.FloatTensor([label])
    g2.y = torch.FloatTensor([label])
    dta_graph[(target, smile)] = [g, g2]
    num_feat_xp = g.x.size()[1]
    num_feat_xd = g2.x.size()[1]

# Main program: iterate over different datasets  and encoding types:
for dataset in datasets:
    print('\nRunning on ', model_st + '_' + dataset)
    df = pd.read_csv('data/'+dataset+'/split/' + dataset + '_train' + setting + '.csv')
    train_drugs, train_prots, train_prots_seq, train_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
        df['target_sequence']), list(df['affinity'])
    train_drugs, train_prots, train_prots_seq, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(
        train_prots_seq), np.asarray(train_Y)

    df = pd.read_csv('data/'+dataset+'/split/' + dataset + '_valid' + setting + '.csv')
    valid_drugs, valid_prots, valid_prots_seq, valid_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
        df['target_sequence']), list(df['affinity'])
    valid_drugs, valid_prots, valid_prots_seq, valid_Y = np.asarray(valid_drugs_drugs), np.asarray(valid_prots), np.asarray(
        valid_prots_seq), np.asarray(valid_Y)

    df = pd.read_csv('data/'+dataset+'/split/' + dataset + '_test' + setting + '.csv')
    test_drugs, test_prots, test_prots_seq, test_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
        df['target_sequence']), list(df['affinity'])
    test_drugs, test_prots, test_prots_seq, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(
        test_prots_seq), np.asarray(test_Y)

    # make data PyTorch Geometric ready
    train_data = GraphPairDataset(smile_list=train_drugs, dta_graph=dta_graph, prot_list=train_prots)
    valid_data = GraphPairDataset(smile_list=valid_drugs, dta_graph=dta_graph, prot_list=valid_prots)
    test_data = GraphPairDataset(smile_list=test_drugs, dta_graph=dta_graph, prot_list=test_prots)
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling(num_features_xd=num_feat_xd,
                     num_features_xt=num_feat_xp,
                     device=device).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'saved_model/'+setting[1:]+'/model_' + model_st + '_' + dataset \
                      + model_name_emb + model_name_seq + model_name_con + model_name_profile \
                      + setting + '.model'
    result_file_name = 'saved_model/'+setting[1:] + '/result_' + model_st + '_' + dataset \
                       + model_name_emb + model_name_seq + model_name_con + model_name_profile \
                       + setting + '.csv'

    # new training
    if from_resume:
        best_mse, best_ci, start_epoch, optimizer, model, LR = resume(model, optimizer, model_file_name)
    else:
        start_epoch = 0
    lr_adjust_patience = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1)
        G, P = predicting(model, device, valid_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        # save the best model based on rmse on validation data
        if set_num == 0:
            if lr_adjust_patience > 40:
                LR = adjust_learning_rate(optimizer, LR, 0.8)
                lr_adjust_patience = 0
        if ret[1] < best_mse:
            best_epoch = epoch + 1
            best_mse = ret[1]
            best_ci = ret[-1]
            G_t, P_t = predicting(model, device, test_loader)
            ret_test = [rmse(G_t, P_t), mse(G_t, P_t), pearson(G_t, P_t), spearman(G_t, P_t), ci(G_t, P_t)]
            # writer.add_scalar('RMSE/test', ret[1], epoch)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret_test)))
            print('RMSE improved at epoch ', best_epoch, '; best_mse, best_ci:',
                  best_mse, best_ci, model_st, dataset, data_path)
            lr_adjust_patience = 0
            save_checkpoint(state={
                'epoch': epoch + 1,
                'best_epoch': best_epoch,
                'arch': model_st,
                'state_dict': model.state_dict(),
                'best_mse': best_mse,
                'best_ci': best_ci,
                'optimizer': optimizer.state_dict(),
                'LR': LR},
                filename=model_file_name
            )
        else:
            print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse, best_ci:',
                  best_mse, best_ci, model_st, dataset, data_path, LR)
            lr_adjust_patience += 1

    # test
    model.load_state_dict(torch.load(model_file_name)['state_dict'], strict=False)
    G, P = predicting(model, device, test_loader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    with open(result_file_name, 'w') as f:
        f.write(','.join(map(str, ret)))
