import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy.interpolate import interp1d

if __name__ == '__main__':

    syn = pd.read_parquet('Data/syn.parquet')
    agro = pd.read_parquet('Data/agro.parquet')

    index = agro.index
    new_syn = pd.DataFrame([], columns=['t2m', 'td2m', 'ff',
                        'R12', 'phi', 'air', 'soilw', 'precip'])
    pbar = tqdm(total=len(list(zip(index[:-1], index[1:]))))

    with pbar:
        for i, j in zip(index[:-1], index[1:]):
            df = agro.loc[[i, j]]
            dur = (df.iloc[1].datetime - df.iloc[0].datetime).days
            if (df.iloc[0].ind == df.iloc[1].ind) & 10 <= dur <= 11:
                ind, t0, t1 = df.iloc[0].ind, df.iloc[0].datetime, df.iloc[1].datetime
                mask = (syn.ind == ind) & (
                    syn.datetime >= t0) & (syn.datetime <= t1)
                data = syn[mask][['t2m', 'td2m', 'ff',
                                'R12', 'phi', 'air', 'soilw', 'precip']]
                new_data = pd.DataFrame([])
                new_data.loc[:, 'ind'] = [ind]*81
                new_data.loc[0, 'datetime'] = t0
                new_data.loc[80, 'datetime'] = t1
                new_data.loc[:, 'datetime'] = new_data.datetime.interpolate(
                    method='ffill')

                n = len(data)
                for col in data.columns:
                    if abs(len(data) - 81) < 9:
                        f = interp1d(np.arange(n), data.loc[:, col].to_numpy())
                        x = np.linspace(0, n-1, 81)
                        new_data.loc[:, col] = f(x)
                    elif abs(len(data) - 81) > 9:
                        f = interp1d(np.arange(n), data.loc[:, col].to_numpy())
                        x = np.linspace(0, n-1, 81)
                new_syn = pd.concat((new_syn, new_data), axis=0)
            pbar.update(1)
            if i > 11 or i < 10:
                continue

    new_syn.astype({'ind': int}).to_parquet('data/new_syn.parquet')
