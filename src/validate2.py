import pandas as pd
import numpy as np
from pandas.core.reshape.concat import concat

from tqdm import tqdm
from scipy.interpolate import interp1d

if __name__ == '__main__':

    syn = pd.read_parquet('Data/syn.parquet')
    agro = pd.read_parquet('Data/agro.parquet')

    index = agro.index
    new_syn = pd.DataFrame([], columns=['t2m', 'td2m', 'ff',
                                        'R12', 'phi', 'air', 'soilw', 'precip'])
    new_agro = pd.DataFrame([], columns=agro.columns)
    pbar = tqdm(total=len(list(zip(index[:-1], index[1:]))))

    with pbar:

        for i, j in zip(index[:-1], index[1:]):

            df = agro.loc[[i, j]]
            dur = (df.iloc[1].datetime - df.iloc[0].datetime).days

            if (df.iloc[0].ind == df.iloc[1].ind) & 10 <= dur <= 11:

                ind, t0, t1 = df.iloc[0].ind, df.iloc[0].datetime, df.iloc[1].datetime
                mask = (syn.ind == ind) & (
                    syn.datetime >= t0) & (syn.datetime <= t1)
                data = syn[mask]

                new_agro = pd.concat((new_agro, df), axis=0)
                new_syn = pd.concat((new_syn, data), axis=0)

            pbar.update(1)

    new_syn.astype({'ind': int}).drop_duplicates().to_parquet('data/new_syn2.parquet')
    new_agro.drop_duplicates().to_parquet('data/new_agro2.parquet')
