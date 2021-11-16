import numpy as np
import pandas as pd
import netCDF4
import argparse 
import logging.config

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from traceback import format_exc
from sklearn.metrics import pairwise_distances

from tools.settings import LOGGING_CONFIG, CLIMATE_OPT, SYN_COLS, AGRO_COLS


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

    А на выходе производная по влагозапасу:

    - val_1
    - val_2
    """, 
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-c', '--climate', action='store_true', help='Использовать ли климатические характеристики')
parser.add_argument('-d', '--dist', type=float, default=100, help='Убрать станции с большим расстоянием')
parser.add_argument('-f', '--fillna', action='store_true', help='Заполнять NAN или нет')
opt = parser.parse_args()

LOGGING_CONFIG['handlers']['file_handler']['filename'] = 'logs/data.log'
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def date_parser(date):
    date = '0'+date if int(date[0]) > 2 else date
    return datetime.strptime(date, '%y%m%d%H')


if __name__ == '__main__':
    
    try:
        logger.info(f'START prepare data with '+', '.join('{}: {}'.format(key, val) for key, val in opt._get_kwargs()))
        
        pathes = list(Path('data/syn').glob('*.csv'))
        
        pbar = tqdm(total=len(pathes)-1, desc='Download syn data')
        syn = pd.read_csv(pathes.pop(), parse_dates=['datetime'], date_parser=date_parser)
        logger.info('Prepare syn data')
        
        with pbar:
            for path in pathes:
                syn = pd.concat((syn, pd.read_csv(path, parse_dates=[
                                'datetime'], date_parser=date_parser)), axis=0)
                pbar.update(1)
        
        syn.loc[syn.R12 != 0, 'R12'] /= 4
        syn.loc[:,'R12'] = syn.R12.interpolate(limit_direction='backward', limitint=3, method='nearest')
        
        if opt.fillna:
            logger.info('fillna')
            syn.loc[:, 'R12'] = syn.R12.fillna(0)
        
        pairs = pd.read_csv('data/pairs/pairs.csv')
        pairs = pairs[pairs.dist < opt.dist]

        syn.loc[:, 'ind'] = syn.s_ind.map(
            {key: val for key, val in pairs[['s_ind', 'ind']].values})
        syn = syn.drop('s_ind', axis=1)
        
        syn.loc[:, 'phi'] = (syn.datetime - syn.datetime.min()).dt.total_seconds()
        syn.loc[:, 'phi'] = 10*np.sin(syn.phi.values/365/24/3600*2*np.pi+0.5)+10

        logger.info('Prepare agro data')
        agro = pd.read_csv('data/agro/agro.csv')
        agro.loc[:, 'datetime'] = pd.to_datetime(agro.year.astype(
            str) + '-' + agro.month.astype(str) + '-' + agro.day.astype(str) + ' 09:00:00')

        dt_min, dt_max = np.max((syn.datetime.min(), agro.datetime.min())), np.min((syn.datetime.max(), agro.datetime.max()))
        syn = syn[(syn.datetime >= dt_min) & (syn.datetime <= dt_max)]
        agro = agro[(agro.datetime >= dt_min) & (agro.datetime <= dt_max)]
        
        #data = syn[SYN_COLS].merge(agro[AGRO_COLS], how='left', on=['ind', 'datetime'])
        agro[AGRO_COLS].sort_values(['ind','datetime']).to_parquet('data/agro.parquet', index=False)

        if opt.climate:

            #TODO: Привести все климатические характеристики к нужному виду, если нужно
            logger.info('Prepare climate data')
            nc = netCDF4.Dataset('data/climate/air.mon.1981-2010.ltm.nc')
            latmask = np.argmin(pairwise_distances(nc['lat'][:].data.reshape(-1, 1),
                                                    pairs['lat'].values.reshape(-1, 1)), axis=0)
            lonmask = np.argmin(pairwise_distances(nc['lon'][:].data.reshape(-1, 1),
                                                    pairs['lon'].values.reshape(-1, 1)), axis=0)
            
            climate = pd.DataFrame()
            
            for i in range(12):
                df = pairs[['ind']].copy()
                for path in CLIMATE_OPT.keys():
                    nc = netCDF4.Dataset(path)
                    df.loc[:, 'month'] = i+1
                    df.loc[:, CLIMATE_OPT[path]] = nc[CLIMATE_OPT[path]][i].data[latmask, lonmask]
                climate = pd.concat((climate,df),axis=0, ignore_index=True)

            syn.loc[:, 'month'] = syn.datetime.dt.month
            syn = syn[SYN_COLS+['month']].merge(climate, on=['ind', 'month']).drop('month', axis=1)
            syn = syn.sort_values(['ind', 'datetime'])

        syn.to_parquet('data/syn.parquet', index=False)
        logger.info('Save syn data')

    except Exception as exp:
        err = format_exc()
        logger.error(err)
        raise(exp)
    
    logger.info('END prepare data')
