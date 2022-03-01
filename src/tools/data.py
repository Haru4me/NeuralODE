"""
Datsets implementation file
"""
from random import shuffle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Data(Dataset):

    def __init__(self, val:bool=False):

        agro = pd.read_parquet('Data/agro.parquet').sort_values(by=['ind', 'datetime'])
        syn = pd.read_parquet('Data/new_syn.parquet').sort_values(by=['ind', 'datetime'])

        self.df = syn.merge(agro, on=['ind', 'datetime']).dropna()
        pairs = self.df[['ind', 'datetime']].drop_duplicates()
        filter = [(i, t0) for i, t0, t1 in zip(pairs.ind, pairs.datetime[:-1], pairs.datetime[1:]) if 10 <= (t1-t0).days <= 11]
        pairs = self.df.groupby(['ind', 'datetime']).count().loc[filter]
        pairs = pairs[pairs.t2m == 80].reset_index()[['ind', 'datetime']]

        self.pairs = [(i, t0, t1) for i, t0, t1 in zip(pairs.ind, pairs.datetime[:-1], pairs.datetime[1:]) if 10 <= (t1-t0).days <= 11]

        train_pairs, val_pairs = train_test_split(self.pairs, test_size=0.1, shuffle=False)

        if val:
            self.pairs = np.random.permutation(val_pairs).tolist()
        else:
            self.pairs = np.random.permutation(train_pairs).tolist()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index:int) -> torch.Tensor:
        i, t1, t2 = self.pairs[index % self.__len__()]
        df = self.df

        data = pd.concat((df[(df.ind == i) & (df.datetime == t1) & (df.datetime < t2)],
                          df[(df.ind == i) & (df.datetime > t1) & (df.datetime == t2)].iloc[[0]]), axis=0)

        y = data[['val_1', 'val_2']].to_numpy()
        X = data[['t2m', 'td2m', 'ff', 'R12', 'phi', 'air', 'soilw', 'precip']].to_numpy()

        return torch.Tensor(X), torch.Tensor(y)
