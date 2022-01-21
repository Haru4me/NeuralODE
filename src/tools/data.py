"""
Datsets implementation file
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):

    def __init__(self):

        # .sort_values(by=['ind', 'datetime'])
        self.agro = pd.read_parquet('Data/agro.parquet')
        # .sort_values(by=['ind', 'datetime'])
        self.syn = pd.read_parquet('Data/new_syn.parquet')

        self.pairs = [(i, t0, t1) for i, t0, t1 in zip(
            self.agro.ind, self.agro.datetime[:-1], self.agro.datetime[1:]) if 10 <= (t1-t0).days <= 11]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index) -> torch.Tensor:
        mask = self.pairs[index % self.__len__()]
        agro_mask = (self.agro.ind == mask[0]) & (self.agro.datetime >=
                                                  mask[1]) & (self.agro.datetime <= mask[2])
        syn_mask = (self.syn.ind == mask[0]) & (self.syn.datetime >=
                                                mask[1]) & (self.syn.datetime <= mask[2])
        y = self.agro[agro_mask][['val_1', 'val_2']].to_numpy()
        X = self.syn[syn_mask][['t2m', 'td2m', 'ff', 'R12',
                                'phi', 'air', 'soilw', 'precip']].to_numpy()[:81]

        return torch.Tensor(X), torch.Tensor(y)
