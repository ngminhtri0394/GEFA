from torch.utils.data import Dataset
import torch

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