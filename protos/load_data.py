import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from logging import getLogger
from multiprocessing import Pool
logger = getLogger(None)

import copy
DROP = ['cnt_dayhouripos', 'row_app', 'uq_channel_ma', 'uq_device_ipchannel', 'row_os', 'uq_os_ipappdevice', 'row_os_device', 'row_app_device', 'uq_app_dayiphourapp', 'row_channel_device', 'row_app_channel', 'row_no_channel',
        'row_channel', 'cnt_nochannel', 'row_os_app', 'row_all', 'uq_device_dayiphourapp', 'row_no_os', 'row_no_app', 'cnt_dayhouripappos', 'row_os_channel', 'row_no_device', 'uq_app_ipdeviceos', 'uq_hour_dayiphourapp']
LIST_ROWS = ['row_ip', 'row_os', 'row_app', 'row_channel', 'row_device', 'row_os_app', 'row_os_channel', 'row_os_device',
             'row_app_channel', 'row_app_device', 'row_channel_device', 'row_no_os', 'row_no_app', 'row_no_channel', 'row_no_device', 'row_all']


def read_csv(path):
    # dtype = copy.deepcopy(DTYPE)
    # if 'test' in path:
    #    dtype['click_id'] = np.int64

    # df = pd.read_csv(path, dtype=dtype, usecols=list(dtype.keys()))
    df = pd.read_csv(path)  # , usecols=LIST_COL)  # [list(dtype.keys())]

    for col in LIST_ROWS:
        if col in df:
            df[col] = df[col] / (df[col] + df[col + '_r'])
            df.drop(col + '_r', axis=1, inplace=True)

    df.drop(['ip', 'hour'] + ['channel'] +
            DROP, axis=1, inplace=True, errors='ignore'
            )

    for col in df:
        if col == 'click_id' and 'test' in path:
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype(np.float32)
    # df = df[(df['span'] > 0)]  # | (df['is_attributed'] == 1)]
    # if 'train' in path:
    #    df = df[(df['day'] == 8)]  # | (df['is_attributed'] == 1)]
    df.drop(['span', 'day'], axis=1, inplace=True)
    return df


def load_train_data():
    paths = sorted(glob.glob('../data/dmt_0422_train/*.csv.gz')) + sorted(glob.glob('../data/dmt_0422_prev/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_valid_data():
    paths = sorted(glob.glob('../data/dmt_0422_valid/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_test_data():
    paths = sorted(glob.glob('../data/dmt_0422_test/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_all_data():
    paths = sorted(glob.glob('../data/dmt_0422_train/*.csv.gz')) + \
        sorted(glob.glob('../data/dmt_0422_valid/*.csv.gz')) + sorted(glob.glob('../data/dmt_0422_prev/*.csv.gz'))

    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


if __name__ == '__main__':
    load_train_data()
    load_test_data()
