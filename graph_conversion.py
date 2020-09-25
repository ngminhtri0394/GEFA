from rdkit import Chem
import networkx as nx
import config
import numpy as np

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    results = one_of_k_encoding_unk(atom.GetSymbol(),
                                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
                                     'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                     'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                                     'Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(),
                                    [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                     Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                     Chem.rdchem.HybridizationType.SP3D2]) + \
              [atom.GetIsAromatic()]
    return np.array(results)


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def aa_features(aa):
    results = one_of_k_encoding(aa,
                                ['A', 'N', 'C', 'Q', 'H', 'L', 'M', 'P', 'T', 'Y', 'R', 'D', 'E', 'G', 'I', 'K', 'F',
                                 'S', 'W', 'V', 'X'])
    return np.asarray(results, dtype=float)


def aa_sas_feature(target, dataset='davis'):
    feature = []
    file = 'data/'+dataset+'/profile/' + target + '_PROP/' + target + '.acc'
    for line in open(file):
        if line[0] == '#':
            continue
        cols = line.strip().split()
        if len(cols) == 6:
            res_sas = []
            res_sas.append(cols[-3])
            res_sas.append(cols[-2])
            res_sas.append(cols[-1])
            feature.append(np.asarray(res_sas, dtype=float))
    return np.asarray(feature)


def aa_ss_feature(target, dataset='davis'):
    feature = []
    file = 'data/'+dataset+'/profile/' + target + '_PROP/' + target + '.ss8'
    for line in open(file):
        cols = line.strip().split()
        if len(cols) == 11:
            res_sas = []
            res_sas.append(cols[-8])
            res_sas.append(cols[-7])
            res_sas.append(cols[-6])
            res_sas.append(cols[-5])
            res_sas.append(cols[-4])
            res_sas.append(cols[-3])
            res_sas.append(cols[-2])
            res_sas.append(cols[-1])
            feature.append(np.asarray(res_sas, dtype=float))
    return np.asarray(feature)


def prot_to_graph(seq, prot_contactmap, prot_target, dataset='davis'):
    c_size = len(seq)
    eds_seq = []
    if config.is_seq_in_graph:
        for i in range(c_size - 1):
            eds_seq.append([i, i + 1])
        eds_seq = np.array(eds_seq)
    eds_contact = []
    if config.is_con_in_graph:
        eds_contact = np.array(np.argwhere(prot_contactmap >= 0.5))

    # add an reserved extra node for drug node
    eds_d = []
    for i in range(c_size):
        eds_d.append([i, c_size])

    eds_d = np.array(eds_d)
    if config.is_seq_in_graph and config.is_con_in_graph:
        eds = np.concatenate((eds_seq, eds_contact, eds_d))
    elif config.is_con_in_graph:
        eds = np.concatenate((eds_contact, eds_d))
    else:
        eds = np.concatenate((eds_seq, eds_d))

    edges = [tuple(i) for i in eds]
    g = nx.Graph(edges).to_directed()
    features = []
    ss_feat = []
    sas_feat = []
    if config.is_profile_in_graph:
        ss_feat = aa_ss_feature(prot_target, dataset)
        sas_feat = aa_sas_feature(prot_target, dataset)
    sequence_output = np.load('data/davis/emb/' + prot_target + '.npz', allow_pickle=True)
    sequence_output = sequence_output[prot_target].reshape(-1, 1)[0][0]['seq'][1:-1, :]
    sequence_output = sequence_output.reshape(sequence_output.shape[0], sequence_output.shape[1])
    for i in range(c_size):
        if config.is_profile_in_graph:
            if config.is_emb_in_graph:
                aa_feat = np.concatenate((np.asarray(sequence_output[i], dtype=float), ss_feat[i], sas_feat[i]))
            else:
                aa_feat = np.concatenate((aa_features(seq[i]), ss_feat[i], sas_feat[i]))
        else:
            if config.is_emb_in_graph:
                aa_feat = np.asarray(sequence_output[i], dtype=float)
            else:
                aa_feat = aa_features(seq[i])
        features.append(aa_feat)

    # place holder feature vector for drug
    place_holder = np.zeros(features[0].shape, dtype=float)
    features.append(place_holder)

    edge_index = []
    edge_weight = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        # if e1 == c_size or e2 == c_size:
        #     edge_weight.append(0.5)
        # else:
        edge_weight.append(1.0)
    return c_size, features, edge_index, edge_weight