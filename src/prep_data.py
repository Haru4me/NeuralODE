import os
import warnings
import argparse
from pathlib import Path
import netCDF4
import pandas as pd
import numpy as np
from geotiff import GeoTiff
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit

from tools.settings import CLIMATE_OPT, CAT_OPT, FEATURES_COLS, START_VAL_COLS, TARGET_COLS

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(prog='Подготовка данных',
    description =
    """
    Скрипт формирует данные для обучения Neural ODE
    По гипотезе на вход подаются:

    - t2m (температура на 2м)
    - td2m (точка росы на 2м)
    - ff (скорость ветра)
    - R (осадки за 6,12,24 часа опционально)
    - phi(t) (периодическая ф-ия по времени)
    - climate (temp, soil, precip) (климатические характеристики температуры, влагозапаса и осадков)
    - soil type (тип почвы)
    - cover type (тип подстилающей поверхности)
    - kult type (тип выращиваемой культуры)
    - val_1, val_2 (ЗПВ на момент времени t0)

    На выходе производная по влагозапасу:

    - new val_1, val_2 (ЗПВ на момент времени t1)
    """,
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-d', '--dist', type=float, default=1, help='Убрать станции с большим расстоянием')
parser.add_argument('-ts', '--test_size', type=float, default=0.1, help='Доля валидационной выборки от общей')
opt = parser.parse_args()

def load_syn(path: str) -> pd.DataFrame:

    syn = pd.read_csv(path, usecols=['s_ind', 'datetime', 't2m', 'td2m', 'ff', 'R12'])
    syn.loc[syn.datetime.astype(str).str.len() == 7, 'datetime'] = '0'+\
        syn[syn.datetime.astype(str).str.len() == 7].datetime.astype(str)
    syn.loc[:, 'datetime'] = pd.to_datetime(syn.datetime, format='%y%m%d%H')

    return syn

def load_agro(path: str) -> pd.DataFrame:

    agro = pd.read_csv(path)
    agro.loc[:,'datetime'] = pd.to_datetime(agro.year.astype(str)+agro.month.astype(str)\
        + agro.day.astype(str)+np.ones(len(agro), dtype='str'), format='%Y%m%d%H', origin='unix')
    agro = agro.drop(['month', 'day'], axis=1)
    agro.loc[:,'prev'] = agro.dec - 1

    return agro

def agro_to_event_period(df: pd.DataFrame) -> pd.DataFrame:

    df = df.merge(df, left_on=['ind', 'dec', 'year'], right_on=['ind', 'prev', 'year'], suffixes=('', '_next'))
    df.loc[:, 'dur'] = (df.datetime_next - df.datetime).dt.days.astype(int)
    df.loc[df.dur == 11, 'datetime_next'] = df[df.dur == 11].datetime_next-pd.Timedelta('1d')
    df.loc[:, 'dur'] = (df.datetime_next - df.datetime).dt.total_seconds().astype(int)

    new_agro = pd.to_datetime((np.repeat(df.datetime.view(int)//int(1e9), 243)\
         + np.hstack([np.arange(0, v, pd.Timedelta('1h').total_seconds()) for v in df.dur+10800.0]))*int(1e9))
    new_agro = df.join(new_agro.rename('ts'), how='outer')

    return new_agro

def data_fusion(agro: pd.DataFrame, syn: pd.DataFrame, pairs: pd.DataFrame, max_dist: float = 50) -> pd.DataFrame:

    syn = syn.merge(pairs[pairs.dist < max_dist], on='s_ind')
    data = agro.merge(syn, left_on=['ind', 'ts'], right_on=['ind','datetime'], how='inner')
    agr = data.groupby(['ind', 'year', 'dec']).val_1.count()
    data = data.set_index(['ind', 'year', 'dec']).loc[agr[agr == 81].index].reset_index()
    data.loc[:, ['t2m', 'td2m', 'ff']] = data[['t2m', 'td2m', 'ff']].interpolate(method='polynomial', order=3)

    for i,j in data[['s_ind','dec']].drop_duplicates().values:
        data.loc[(data.s_ind == i) & (data.dec == j), 'R12'] = \
            (data[(data.s_ind == i) & (data.dec == j)].R12/4).fillna(method='bfill', limit=3).fillna(0)

    return data

def load_climate(optinons: dict, pairs: pd.DataFrame) -> pd.DataFrame:

    path = list(optinons.keys())[0]
    nc = netCDF4.Dataset(path)

    latmask = np.argmin(pairwise_distances(nc['lat'][:].data.reshape(-1, 1),
                                        pairs['s_lat'].values.reshape(-1, 1)), axis=0)
    lonmask = np.argmin(pairwise_distances(nc['lon'][:].data.reshape(-1, 1),
                                        pairs['s_lon'].values.reshape(-1, 1)), axis=0)

    climate = pd.DataFrame()

    for i in range(12):

        df = pairs[['s_ind']].copy()

        for path in optinons.keys():

            nc = netCDF4.Dataset(path)
            df.loc[:, 'month'] = i+1
            df.loc[:, optinons[path]] = nc[optinons[path]][i].data[latmask, lonmask]

        climate = pd.concat((climate, df), axis=0, ignore_index=True)

    return climate.drop_duplicates()

def decode_tif(lat: np.array, lon: np.array, tifname: str) -> np.array:

    lon1 = lon.min()
    lon2 = lon.max()
    lat1 = lat.min()
    lat2 = lat.max()
    arr = np.array(GeoTiff(tifname).read_box([(lon1, lat1), (lon2, lat2)]))
    ilon = np.round((lon-lon1)/(lon2-lon1)*(arr.shape[1]-1)).round().astype(np.int64)
    ilat = np.round((lat2-lat)/(lat2-lat1)*(arr.shape[0]-1)).round().astype(np.int64)
    out = np.array([arr[ilat[i], ilon[i]] for i in range(ilon.shape[0])])

    return out

def load_soil_cats(pathes: list, pairs: pd.DataFrame) -> pd.DataFrame:

    lat, lon = pairs.loc[:, 'lat'].to_numpy().astype(int), pairs.loc[:, 'lon'].to_numpy().astype(int)
    pairs.loc[:, 'soiltype'] = decode_tif(lat, lon, pathes['soil']['tiff'])
    pairs.loc[:, 'covertype'] = decode_tif(lat, lon, pathes['cover']['tiff'])

    soil_df = pd.read_csv(pathes['soil']['description'], sep='\t')
    cover_df = pd.read_excel(pathes['cover']['description'], usecols=['Value', 'Label'])

    soils = pairs.merge(cover_df, left_on='covertype', right_on='Value')\
               .merge(soil_df, left_on='soiltype', right_on='GRIDCODE')\
               .drop(['Value', 'GRIDCODE', 'lat', 'lon', 's_ind', 'dist', 's_lat', 's_lon'], axis=1)\
               .rename(columns={'Label': 'cover_name', 'SOIL_ORDER': 'soil_label', 'SUBORDER': 'suborder'})\
               .astype({'covertype':'int64'})

    soils.loc[:, 'covertype'] = soils.covertype.map(
        {elm: i for i, elm in enumerate(soils.covertype.sort_values().unique())}).astype(int)

    soils.loc[:, 'soiltype'] = soils.soiltype.map(
        {elm: i for i, elm in enumerate(soils.soiltype.sort_values().unique())}).astype(int)

    soils_label = pd.DataFrame()
    soils_label.loc[:, 'soiltypes'] = {i: elm for i, elm in  enumerate(soils.soil_label.unique())}.keys()
    soils_label.loc[:, 'soil_label'] = {i: elm for i, elm in  enumerate(soils.soil_label.unique())}.values()

    soils = soils.merge(soils_label, on='soil_label')\
                    .drop('soiltype', axis=1)\
                    .rename(columns={'soiltypes': 'soiltype'})

    return soils

def save_to_npz(data: pd.DataFrame, features: list, start: list, target: list, test_size: float = 0.1) -> None:

    k = True

    while k:

        ind = np.random.choice(data.ind.unique())
        year = np.random.choice(data.ts.dt.year.unique())
        n = data[(data.ind == ind) & (data.ts.dt.year == year)].dec.nunique()
        k = (n > 16) or (n < 11)

    data = data.set_index(['ind', 'year', 'dec'])
    sample_idx = data.loc[[ind], [year], :].index.unique().to_numpy()
    tv_data = data.drop(sample_idx)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(gss.split(X=tv_data, y=tv_data[['val_1_next', 'val_2_next']], groups=tv_data.reset_index()['year']))
    train_idx, val_idx = np.unique(tv_data.index.to_numpy()[train_idx]), np.unique(tv_data.index.to_numpy()[val_idx])
    all_idx = {'train': train_idx, 'val': val_idx, 'sample': sample_idx}

    for key in all_idx.keys():

        for ind, year, dec in tqdm(all_idx[key], desc=f'Saving {key} to npz'):

            v = data.loc[ind, year, dec][features].to_numpy()
            z0 = data.loc[ind, year, dec][start].to_numpy()[0]
            z1 = data.loc[ind, year, dec][target].to_numpy()[0]

            np.savez_compressed(f'data/dataset/{key}/{ind}_{year}_{dec}.npz',
                                v=v, z0=z0, z1=z1, ind=ind, year=year, dec=dec)

    alls = set(Path('data/dataset').rglob('*.npz'))

    for path in tqdm(alls, desc="Search data with NaN"):

        file = np.load(path)
        v, z0, z1 = file['v'], file['z0'], file['z1']

        if np.isnan(v).sum() or np.isnan(z0).sum() or np.isnan(z1).sum():
            os.remove(path)

def clear_syn(syn: pd.DataFrame):

    syn.R12[syn.R12 == 9990] = 0.1
    syn = syn[syn.t2m.abs() < 60]
    syn = syn[syn.td2m.abs() < 60]
    syn = syn[syn.ff <= 30]

    return syn

def cat_prep(data: pd.DataFrame):

    cover_frac = data[['cover_name']].value_counts().reset_index().rename(columns={0:'perc'})
    cover_frac.loc[:, 'perc'] = cover_frac.perc/cover_frac.perc.sum()*100
    cover_frac.loc[:, 'cover_name_new'] = cover_frac.cover_name
    cover_frac.loc[cover_frac.perc < 5, 'cover_name_new'] = 'Other'
    cover_frac = cover_frac.drop(['perc'], axis=1)

    soil_frac = data[['soil_label']].value_counts().reset_index().rename(columns={0:'perc'})
    soil_frac.loc[:, 'perc'] = soil_frac.perc/soil_frac.perc.sum()*100
    soil_frac.loc[:, 'soil_label_new'] = soil_frac.soil_label
    soil_frac.loc[soil_frac.perc < 2, 'soil_label_new'] = 'Other'
    soil_frac = soil_frac.drop(['perc'], axis=1)

    cult = pd.read_csv('data/agro/cult.csv', sep=';').rename(columns={'id': 'kult'})
    data = data.merge(cover_frac, on='cover_name')\
                .merge(soil_frac, on='soil_label')\
                .merge(cult, on='kult')\
                .drop(['cover_name', 'soil_label'], axis=1)\
                .rename(columns={'cover_name_new': 'cover_name', 'soil_label_new': 'soil_label'})

    data.loc[:, 'soiltype'] = data.soil_label.map({elm: i for i,elm in enumerate(data.soil_label.unique())})
    data.loc[:, 'covertype'] = data.cover_name.map({elm: i for i,elm in enumerate(data.cover_name.unique())})
    data.loc[:, 'culttype'] = data.type.map({elm: i for i,elm in enumerate(data.type.unique())})

    return data

if __name__ == '__main__':

    paths = {
        'agro': 'data/agro/agro.csv',
        'pairs': 'data/pairs/pairs.csv',
        'syn': list(Path('data/syn').rglob('*.csv'))
    }

    agro = load_agro(paths['agro'])
    agro = agro_to_event_period(agro)
    pairs = pd.read_csv(paths['pairs'])
    climate = load_climate(CLIMATE_OPT, pairs.copy())
    soil = load_soil_cats(CAT_OPT, pairs.copy())

    syn = pd.concat([load_syn(file) for file in tqdm(paths['syn'], desc='Load synoptical data')], axis=0)
    syn = clear_syn(syn.copy())

    data = data_fusion(agro.copy(), syn.copy(), pairs.copy(), max_dist=opt.dist)
    data = data.merge(climate, left_on=['s_ind', data.ts.dt.month], right_on=['s_ind','month'])\
               .merge(soil, on='ind')
    data.loc[:, 'phi'] = np.sin(((data.ts-pd.Timestamp('1970-01-01'))/pd.Timedelta(seconds=1)/pd.Timedelta(days=365.24).total_seconds()*2*np.pi))
    data.loc[:, 'air'] = data.air-272.1
    data = cat_prep(data.copy())
    data = data.sort_values(['ind','year', 'dec'])
    data.to_parquet('data/data.pq')

    alls = set(Path('data/dataset').rglob('*.npz'))

    for path in tqdm(alls, desc="Delete old data"):
        os.remove(path)

    save_to_npz(data, FEATURES_COLS, START_VAL_COLS, TARGET_COLS, test_size=opt.test_size)
